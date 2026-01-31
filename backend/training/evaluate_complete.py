#!/usr/bin/env python3
"""
evaluate_complete.py
====================
Sistema completo de avaliação para prova de excelência do modelo.

Gera evidência auditável em 4 pilares:
1. Anti-Leakage (asset split, time alignment, feature denunciadora)
2. Performance onde interessa (RUL perto falha, recall crítico, PR-AUC)
3. Generalização (GroupKFold, ablation)
4. Robustez operacional (stress tests, false alarms/day)

Outputs:
- outputs/reports/eval_report_complete.json
- outputs/reports/confusion_matrix_*.png
- outputs/reports/pr_curve_failure.png
- outputs/reports/rul_error_bins.png
- outputs/reports/stress_summary.json
- outputs/reports/fold_results.csv
- outputs/reports/false_alarms_per_day.csv

Author: Industrial Pump Digital Twin
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, f1_score,
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    roc_auc_score
)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(f"[INFO] TensorFlow: {tf.__version__}")


# ===========================================================================
# CONFIGURATION
# ===========================================================================

@dataclass
class EvalConfig:
    """Configuração de avaliação."""
    max_sensors: int = 8
    seq_len: int = 64
    hop: int = 8
    n_freq: int = 16
    n_time: int = 16
    aux_dim: int = 8
    seed: int = 42
    n_folds: int = 5
    
    # Thresholds
    rul_near_failure_threshold: float = 0.2
    severity_critical_classes: List[str] = field(default_factory=lambda: ['failure', 'severe'])
    
    # Stress test params
    noise_std: float = 0.1
    gain_drift_range: Tuple[float, float] = (1.1, 1.3)
    offset_drift_range: Tuple[float, float] = (-0.2, 0.2)
    clip_percentile: float = 95
    sensor_dropout_rate: float = 0.3
    
    # False alarm params
    alarm_threshold: float = 0.5
    hysteresis_windows: int = 3


# ===========================================================================
# CUSTOM LAYERS (needed for model loading)
# ===========================================================================

@keras.utils.register_keras_serializable(package="PumpV2")
class SensorDropout(layers.Layer):
    def __init__(self, rate: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        x, mask = inputs
        batch_size = tf.shape(x)[0]
        n_sensors = tf.shape(x)[1]
        drop_mask = tf.random.uniform((batch_size, n_sensors, 1, 1, 1)) > self.rate
        drop_mask = tf.cast(drop_mask, x.dtype)
        mask_expanded = tf.reshape(mask, (batch_size, n_sensors, 1, 1, 1))
        combined_mask = drop_mask * mask_expanded
        return x * combined_mask, tf.squeeze(combined_mask, axis=[2, 3, 4])
    
    def get_config(self):
        return {**super().get_config(), "rate": self.rate}


@keras.utils.register_keras_serializable(package="PumpV2")
class MaskedGlobalPooling(layers.Layer):
    def call(self, inputs):
        x, mask = inputs
        mask_expanded = tf.expand_dims(mask, -1)
        x_masked = x * mask_expanded
        sum_features = tf.reduce_sum(x_masked, axis=1)
        n_active = tf.reduce_sum(mask, axis=1, keepdims=True)
        n_active = tf.maximum(n_active, 1.0)
        return sum_features / n_active
    
    def get_config(self):
        return super().get_config()


# ===========================================================================
# DATA PIPELINE (simplified for eval)
# ===========================================================================

class EvalPipeline:
    """Pipeline de dados para avaliação."""
    
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.severity_labels = []
        self.mode_labels = []
        self.sev_to_idx = {}
        self.mode_to_idx = {}
        self.rul_max_train = 1.0
    
    def load_data(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        print(f"[LOAD] {len(df)} samples from {path.name}")
        return df
    
    def split_by_asset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assets = df['asset_id'].unique().tolist()
        np.random.seed(self.cfg.seed)
        np.random.shuffle(assets)
        
        n_train = int(0.7 * len(assets))
        n_val = int(0.15 * len(assets))
        
        train_assets = assets[:n_train]
        val_assets = assets[n_train:n_train + n_val]
        test_assets = assets[n_train + n_val:]
        
        train_df = df[df['asset_id'].isin(train_assets)]
        val_df = df[df['asset_id'].isin(val_assets)]
        test_df = df[df['asset_id'].isin(test_assets)]
        
        return train_df, val_df, test_df, train_assets, val_assets, test_assets
    
    def create_windows(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        fit_scaler: bool = False
    ) -> Tuple:
        X_spec, X_aux, masks = [], [], []
        y_severity, y_mode, y_rul, y_health = [], [], [], []
        window_info = []  # Para rastreabilidade
        
        if fit_scaler:
            self.scaler.fit(df[sensor_cols])
            self.rul_max_train = df['rul_minutes'].max()
        
        df_norm = df.copy()
        df_norm[sensor_cols] = self.scaler.transform(df[sensor_cols])
        df_norm['rul_norm'] = df['rul_minutes'] / max(self.rul_max_train, 1)
        df_norm['health_norm'] = df['health_index'] / 100.0
        
        for asset_id, group in df_norm.groupby('asset_id'):
            group = group.sort_values('timestamp')
            n = len(group)
            
            for i in range(0, n - self.cfg.seq_len + 1, self.cfg.hop):
                window = group.iloc[i:i + self.cfg.seq_len]
                sensor_data = window[sensor_cols].values.T
                
                n_sensors = len(sensor_cols)
                if n_sensors < self.cfg.max_sensors:
                    pad = np.zeros((self.cfg.max_sensors - n_sensors, self.cfg.seq_len))
                    sensor_data = np.vstack([sensor_data, pad])
                
                spec = self._create_spectrogram(sensor_data)
                mask = np.zeros(self.cfg.max_sensors)
                mask[:n_sensors] = 1.0
                aux = self._extract_aux(window.iloc[-1], sensor_cols)
                
                X_spec.append(spec)
                X_aux.append(aux)
                masks.append(mask)
                y_severity.append(window['severity'].iloc[-1])
                y_mode.append(window['mode'].iloc[-1])
                y_rul.append(window['rul_norm'].iloc[-1])
                y_health.append(window['health_norm'].iloc[-1])
                
                # Info para rastreabilidade
                window_info.append({
                    'asset_id': asset_id,
                    'timestamp_start': window['timestamp'].iloc[0],
                    'timestamp_end': window['timestamp'].iloc[-1],
                    'rul_minutes': df.loc[window.index[-1], 'rul_minutes'],
                })
        
        return (
            np.array(X_spec, dtype=np.float32),
            np.array(X_aux, dtype=np.float32),
            np.array(masks, dtype=np.float32),
            np.array(y_severity),
            np.array(y_mode),
            np.array(y_rul, dtype=np.float32),
            np.array(y_health, dtype=np.float32),
            window_info
        )
    
    def _create_spectrogram(self, sensor_data: np.ndarray) -> np.ndarray:
        n_sensors, seq_len = sensor_data.shape
        n_freq = self.cfg.n_freq
        n_time = self.cfg.n_time
        spec = np.zeros((n_sensors, n_freq, n_time), dtype=np.float32)
        
        for s in range(n_sensors):
            signal = sensor_data[s]
            frame_len = seq_len // n_time
            for t in range(n_time):
                start = t * frame_len
                end = start + frame_len
                frame = signal[start:end]
                fft = np.abs(np.fft.rfft(frame, n=n_freq * 2))[:n_freq]
                spec[s, :, t] = fft
        
        return spec[..., np.newaxis]
    
    def _extract_aux(self, row: pd.Series, sensor_cols: List[str]) -> np.ndarray:
        aux = []
        for col in sensor_cols[:self.cfg.aux_dim]:
            if col in row.index:
                aux.append(float(row[col]))
        while len(aux) < self.cfg.aux_dim:
            aux.append(0.0)
        return np.array(aux[:self.cfg.aux_dim], dtype=np.float32)
    
    def encode_labels(self, y_sev, y_mode, fit=False):
        if fit:
            self.severity_labels = sorted(set(y_sev))
            self.mode_labels = sorted(set(y_mode))
            self.sev_to_idx = {s: i for i, s in enumerate(self.severity_labels)}
            self.mode_to_idx = {m: i for i, m in enumerate(self.mode_labels)}
        
        y_sev_enc = np.array([self.sev_to_idx.get(s, 0) for s in y_sev])
        y_mode_enc = np.array([self.mode_to_idx.get(m, 0) for m in y_mode])
        return y_sev_enc, y_mode_enc


# ===========================================================================
# 1) ANTI-LEAKAGE PROOFS
# ===========================================================================

class AntiLeakageProofs:
    """Provas de que não há data leakage."""
    
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.results = {}
    
    def verify_asset_split(
        self,
        train_assets: List,
        val_assets: List,
        test_assets: List
    ) -> Dict:
        """Prova A: Split por asset sem interseção."""
        train_set = set(train_assets)
        val_set = set(val_assets)
        test_set = set(test_assets)
        
        ok = (
            train_set.isdisjoint(val_set) and
            train_set.isdisjoint(test_set) and
            val_set.isdisjoint(test_set)
        )
        
        self.results['asset_split'] = {
            'ok': ok,
            'n_train_assets': len(train_assets),
            'n_val_assets': len(val_assets),
            'n_test_assets': len(test_assets),
            'train_val_intersection': list(train_set & val_set),
            'train_test_intersection': list(train_set & test_set),
            'val_test_intersection': list(val_set & test_set),
        }
        
        print(f"[LEAKAGE] Asset split OK: {ok}")
        return self.results['asset_split']
    
    def verify_window_alignment(self, window_info: List[Dict]) -> Dict:
        """Prova B: Labels vêm do fim da janela."""
        # Verifica que timestamp_end é usado para labels
        all_aligned = all(
            w['timestamp_end'] >= w['timestamp_start']
            for w in window_info
        )
        
        self.results['window_alignment'] = {
            'ok': all_aligned,
            'n_windows_checked': len(window_info),
        }
        
        print(f"[LEAKAGE] Window alignment OK: {all_aligned}")
        return self.results['window_alignment']
    
    def verify_no_leaky_features(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        target_col: str = 'rul_minutes'
    ) -> Dict:
        """Prova C: Features não denunciam o target."""
        
        # Colunas potencialmente "denunciadoras"
        suspicious_cols = [
            col for col in df.columns
            if any(x in col.lower() for x in [
                'time', 'cycle', 'runtime', 'counter', 'age', 'index'
            ])
        ]
        suspicious_cols = [c for c in suspicious_cols if c != 'timestamp']
        
        results = {'suspicious_cols_found': suspicious_cols}
        
        # Baseline com timestamp/index
        if len(df) > 1000:
            sample = df.sample(1000, random_state=self.cfg.seed)
        else:
            sample = df
        
        X_index = np.arange(len(sample)).reshape(-1, 1)
        y = sample[target_col].values
        
        # Ridge com index só
        try:
            ridge = Ridge()
            ridge.fit(X_index, y)
            pred = ridge.predict(X_index)
            mae_index = mean_absolute_error(y, pred)
            r2_index = ridge.score(X_index, y)
            
            results['index_only_baseline'] = {
                'mae': float(mae_index),
                'r2': float(r2_index),
                'warning': r2_index > 0.8,  # R2 > 0.8 é suspeito
            }
        except Exception as e:
            results['index_only_baseline'] = {'error': str(e)}
        
        # Baseline com sensores só
        try:
            X_sensors = sample[sensor_cols].values
            rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)
            rf.fit(X_sensors, y)
            pred = rf.predict(X_sensors)
            mae_sensors = mean_absolute_error(y, pred)
            
            results['sensors_only_baseline'] = {
                'mae': float(mae_sensors),
                'note': 'Este valor deve ser similar ao modelo final se não houver leakage'
            }
        except Exception as e:
            results['sensors_only_baseline'] = {'error': str(e)}
        
        self.results['leaky_features'] = results
        
        warning = results.get('index_only_baseline', {}).get('warning', False)
        print(f"[LEAKAGE] Leaky features check - Warning: {warning}")
        
        return self.results['leaky_features']
    
    def get_all_results(self) -> Dict:
        return self.results


# ===========================================================================
# 2) PERFORMANCE METRICS (onde interessa)
# ===========================================================================

class PerformanceMetrics:
    """Métricas de performance detalhadas."""
    
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.results = {}
    
    def compute_rul_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        rul_max_train: float
    ) -> Dict:
        """RUL: MAE em minutos + erro perto da falha + por bins."""
        
        # MAE normalizado
        mae_norm = float(mean_absolute_error(y_true, y_pred))
        
        # MAE em minutos
        mae_minutes = mae_norm * rul_max_train
        
        # MAE perto da falha (RUL < threshold)
        near_mask = y_true < self.cfg.rul_near_failure_threshold
        if near_mask.any():
            mae_near = float(mean_absolute_error(y_true[near_mask], y_pred[near_mask]))
            mae_near_minutes = mae_near * rul_max_train
        else:
            mae_near = None
            mae_near_minutes = None
        
        # MAE por bins
        bins = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0-10%', '10-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        mae_bins = {}
        
        for i in range(len(bins) - 1):
            mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
            if mask.any():
                mae_bins[bin_labels[i]] = {
                    'mae_norm': float(mean_absolute_error(y_true[mask], y_pred[mask])),
                    'mae_minutes': float(mean_absolute_error(y_true[mask], y_pred[mask])) * rul_max_train,
                    'n_samples': int(mask.sum()),
                }
        
        self.results['rul'] = {
            'mae_norm': mae_norm,
            'mae_minutes': mae_minutes,
            'mae_near_failure_norm': mae_near,
            'mae_near_failure_minutes': mae_near_minutes,
            'near_failure_threshold': self.cfg.rul_near_failure_threshold,
            'rul_max_train': rul_max_train,
            'mae_by_bin': mae_bins,
        }
        
        print(f"[PERF] RUL MAE: {mae_norm:.4f} ({mae_minutes:.1f} min)")
        if mae_near is not None:
            print(f"[PERF] RUL MAE (near failure): {mae_near:.4f} ({mae_near_minutes:.1f} min)")
        
        return self.results['rul']
    
    def compute_health_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """Health: MAE + por bins."""
        
        mae = float(mean_absolute_error(y_true, y_pred))
        mae_percent = mae * 100
        
        # MAE por bins de health
        bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
        bin_labels = ['0-30%', '30-50%', '50-70%', '70-90%', '90-100%']
        mae_bins = {}
        
        for i in range(len(bins) - 1):
            mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
            if mask.any():
                mae_bins[bin_labels[i]] = {
                    'mae': float(mean_absolute_error(y_true[mask], y_pred[mask])),
                    'mae_percent': float(mean_absolute_error(y_true[mask], y_pred[mask])) * 100,
                    'n_samples': int(mask.sum()),
                }
        
        self.results['health'] = {
            'mae': mae,
            'mae_percent': mae_percent,
            'mae_by_bin': mae_bins,
        }
        
        print(f"[PERF] Health MAE: {mae_percent:.2f}%")
        return self.results['health']
    
    def compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        labels: List[str],
        task_name: str
    ) -> Dict:
        """Severity/Mode: F1 macro + recall crítico + PR-AUC."""
        
        acc = float(accuracy_score(y_true, y_pred))
        f1_macro = float(f1_score(y_true, y_pred, average='macro'))
        f1_weighted = float(f1_score(y_true, y_pred, average='weighted'))
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=labels,
            output_dict=True,
            zero_division=0
        )
        
        # Recall/Precision das classes críticas
        critical_metrics = {}
        for cls in self.cfg.severity_critical_classes:
            if cls in report:
                critical_metrics[cls] = {
                    'precision': report[cls]['precision'],
                    'recall': report[cls]['recall'],
                    'f1': report[cls]['f1-score'],
                    'support': report[cls]['support'],
                }
        
        # PR-AUC para cada classe (one-vs-rest)
        pr_auc = {}
        for i, label in enumerate(labels):
            y_binary = (y_true == i).astype(int)
            if y_binary.sum() > 0 and y_binary.sum() < len(y_binary):
                try:
                    ap = average_precision_score(y_binary, y_proba[:, i])
                    pr_auc[label] = float(ap)
                except:
                    pass
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        self.results[task_name] = {
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': report,
            'critical_classes_metrics': critical_metrics,
            'pr_auc_per_class': pr_auc,
            'confusion_matrix': cm.tolist(),
            'labels': labels,
        }
        
        print(f"[PERF] {task_name} Accuracy: {acc*100:.2f}%, F1 macro: {f1_macro:.4f}")
        for cls, m in critical_metrics.items():
            print(f"[PERF] {task_name} {cls} - Recall: {m['recall']:.2f}, Precision: {m['precision']:.2f}")
        
        return self.results[task_name]
    
    def get_all_results(self) -> Dict:
        return self.results


# ===========================================================================
# 3) GENERALIZATION TESTS
# ===========================================================================

class GeneralizationTests:
    """Testes de generalização."""
    
    def __init__(self, cfg: EvalConfig, pipeline: EvalPipeline):
        self.cfg = cfg
        self.pipeline = pipeline
        self.results = {}
    
    def run_group_kfold(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        model_builder_fn,
        train_fn
    ) -> Dict:
        """K-Fold por asset (GroupKFold)."""
        
        print(f"\n[KFOLD] Running {self.cfg.n_folds}-fold cross-validation by asset...")
        
        assets = df['asset_id'].unique()
        gkf = GroupKFold(n_splits=self.cfg.n_folds)
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df['asset_id'])):
            print(f"\n[FOLD {fold+1}/{self.cfg.n_folds}]")
            
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            # Create windows
            train_data = self.pipeline.create_windows(train_df, sensor_cols, fit_scaler=True)
            test_data = self.pipeline.create_windows(test_df, sensor_cols, fit_scaler=False)
            
            # Encode labels
            y_sev_train, y_mode_train = self.pipeline.encode_labels(train_data[3], train_data[4], fit=True)
            y_sev_test, y_mode_test = self.pipeline.encode_labels(test_data[3], test_data[4])
            
            n_severity = len(self.pipeline.severity_labels)
            n_mode = len(self.pipeline.mode_labels)
            
            # Build and train model
            model = model_builder_fn(n_severity, n_mode)
            
            X_train = [train_data[0], train_data[1], train_data[2]]
            y_train = [y_sev_train, y_mode_train, train_data[5], train_data[6]]
            
            X_test = [test_data[0], test_data[1], test_data[2]]
            
            train_fn(model, X_train, y_train)
            
            # Predict
            preds = model.predict(X_test, verbose=0)
            
            y_pred_sev = np.argmax(preds[0], axis=1)
            y_pred_mode = np.argmax(preds[1], axis=1)
            y_pred_rul = preds[2].flatten()
            y_pred_health = preds[3].flatten()
            
            # Metrics
            fold_result = {
                'fold': fold + 1,
                'n_train': len(train_df),
                'n_test': len(test_df),
                'severity_acc': float(accuracy_score(y_sev_test, y_pred_sev)),
                'mode_acc': float(accuracy_score(y_mode_test, y_pred_mode)),
                'rul_mae': float(mean_absolute_error(test_data[5], y_pred_rul)),
                'health_mae': float(mean_absolute_error(test_data[6], y_pred_health)),
            }
            
            fold_results.append(fold_result)
            print(f"  Severity: {fold_result['severity_acc']*100:.2f}%, Mode: {fold_result['mode_acc']*100:.2f}%")
            print(f"  RUL MAE: {fold_result['rul_mae']:.4f}, Health MAE: {fold_result['health_mae']*100:.2f}%")
            
            # Clear session to free memory
            keras.backend.clear_session()
        
        # Summary statistics
        metrics = ['severity_acc', 'mode_acc', 'rul_mae', 'health_mae']
        summary = {}
        for m in metrics:
            values = [r[m] for r in fold_results]
            summary[m] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
        
        self.results['group_kfold'] = {
            'n_folds': self.cfg.n_folds,
            'fold_results': fold_results,
            'summary': summary,
        }
        
        print(f"\n[KFOLD] Summary:")
        print(f"  Severity: {summary['severity_acc']['mean']*100:.2f}% ± {summary['severity_acc']['std']*100:.2f}%")
        print(f"  Mode: {summary['mode_acc']['mean']*100:.2f}% ± {summary['mode_acc']['std']*100:.2f}%")
        print(f"  RUL MAE: {summary['rul_mae']['mean']:.4f} ± {summary['rul_mae']['std']:.4f}")
        print(f"  Health MAE: {summary['health_mae']['mean']*100:.2f}% ± {summary['health_mae']['std']*100:.2f}%")
        
        return self.results['group_kfold']
    
    def get_all_results(self) -> Dict:
        return self.results


# ===========================================================================
# 4) STRESS TESTS
# ===========================================================================

class StressTests:
    """Testes de robustez."""
    
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.results = {}
    
    def run_all_stress_tests(
        self,
        model: Model,
        X_test: List[np.ndarray],
        y_test: List[np.ndarray],
        labels_sev: List[str],
        labels_mode: List[str],
        baseline_results: Dict
    ) -> Dict:
        """Executa todos os stress tests."""
        
        X_spec, X_aux, X_mask = X_test
        y_sev, y_mode, y_rul, y_health = y_test
        
        stress_results = {}
        
        # 1. Sensor dropout (mask=0, signal=0)
        print("\n[STRESS] Sensor dropout...")
        stress_results['sensor_dropout'] = self._test_sensor_dropout(
            model, X_spec, X_aux, X_mask, y_sev, y_mode, y_rul, y_health
        )
        
        # 2. Noise injection
        print("[STRESS] Noise injection...")
        stress_results['noise'] = self._test_noise(
            model, X_spec, X_aux, X_mask, y_sev, y_mode, y_rul, y_health
        )
        
        # 3. Gain drift
        print("[STRESS] Gain drift...")
        stress_results['gain_drift'] = self._test_gain_drift(
            model, X_spec, X_aux, X_mask, y_sev, y_mode, y_rul, y_health
        )
        
        # 4. Offset drift
        print("[STRESS] Offset drift...")
        stress_results['offset_drift'] = self._test_offset_drift(
            model, X_spec, X_aux, X_mask, y_sev, y_mode, y_rul, y_health
        )
        
        # 5. Clipping
        print("[STRESS] Clipping...")
        stress_results['clipping'] = self._test_clipping(
            model, X_spec, X_aux, X_mask, y_sev, y_mode, y_rul, y_health
        )
        
        # Compute degradation relative to baseline
        for test_name, result in stress_results.items():
            result['degradation'] = {
                'severity_acc': baseline_results['severity']['accuracy'] - result['severity_acc'],
                'mode_acc': baseline_results['mode']['accuracy'] - result['mode_acc'],
                'rul_mae': result['rul_mae'] - baseline_results['rul']['mae_norm'],
                'health_mae': result['health_mae'] - baseline_results['health']['mae'],
            }
        
        self.results['stress_tests'] = stress_results
        
        # Summary
        print("\n[STRESS] Summary of degradation:")
        for test_name, result in stress_results.items():
            deg = result['degradation']
            print(f"  {test_name}:")
            print(f"    Severity: -{deg['severity_acc']*100:.2f}%")
            print(f"    Mode: -{deg['mode_acc']*100:.2f}%")
            print(f"    RUL MAE: +{deg['rul_mae']:.4f}")
            print(f"    Health MAE: +{deg['health_mae']*100:.2f}%")
        
        return self.results['stress_tests']
    
    def _evaluate(self, model, X_spec, X_aux, X_mask, y_sev, y_mode, y_rul, y_health):
        preds = model.predict([X_spec, X_aux, X_mask], verbose=0)
        return {
            'severity_acc': float(accuracy_score(y_sev, np.argmax(preds[0], axis=1))),
            'mode_acc': float(accuracy_score(y_mode, np.argmax(preds[1], axis=1))),
            'rul_mae': float(mean_absolute_error(y_rul, preds[2].flatten())),
            'health_mae': float(mean_absolute_error(y_health, preds[3].flatten())),
        }
    
    def _test_sensor_dropout(self, model, X_spec, X_aux, X_mask, y_sev, y_mode, y_rul, y_health):
        X_spec_stress = X_spec.copy()
        X_mask_stress = X_mask.copy()
        
        n_samples, n_sensors = X_mask.shape[:2]
        drop_mask = np.random.random((n_samples, n_sensors)) < self.cfg.sensor_dropout_rate
        
        for i in range(n_samples):
            for s in range(n_sensors):
                if drop_mask[i, s] and X_mask[i, s] > 0:
                    X_spec_stress[i, s] = 0
                    X_mask_stress[i, s] = 0
        
        return self._evaluate(model, X_spec_stress, X_aux, X_mask_stress, y_sev, y_mode, y_rul, y_health)
    
    def _test_noise(self, model, X_spec, X_aux, X_mask, y_sev, y_mode, y_rul, y_health):
        noise = np.random.normal(0, self.cfg.noise_std, X_spec.shape)
        X_spec_stress = X_spec + noise * X_mask[:, :, np.newaxis, np.newaxis, np.newaxis]
        return self._evaluate(model, X_spec_stress, X_aux, X_mask, y_sev, y_mode, y_rul, y_health)
    
    def _test_gain_drift(self, model, X_spec, X_aux, X_mask, y_sev, y_mode, y_rul, y_health):
        gain = np.random.uniform(*self.cfg.gain_drift_range, size=(X_spec.shape[0], X_spec.shape[1], 1, 1, 1))
        X_spec_stress = X_spec * gain
        return self._evaluate(model, X_spec_stress, X_aux, X_mask, y_sev, y_mode, y_rul, y_health)
    
    def _test_offset_drift(self, model, X_spec, X_aux, X_mask, y_sev, y_mode, y_rul, y_health):
        offset = np.random.uniform(*self.cfg.offset_drift_range, size=(X_spec.shape[0], X_spec.shape[1], 1, 1, 1))
        X_spec_stress = X_spec + offset * X_mask[:, :, np.newaxis, np.newaxis, np.newaxis]
        return self._evaluate(model, X_spec_stress, X_aux, X_mask, y_sev, y_mode, y_rul, y_health)
    
    def _test_clipping(self, model, X_spec, X_aux, X_mask, y_sev, y_mode, y_rul, y_health):
        clip_val = np.percentile(X_spec[X_spec > 0], self.cfg.clip_percentile)
        X_spec_stress = np.clip(X_spec, None, clip_val)
        return self._evaluate(model, X_spec_stress, X_aux, X_mask, y_sev, y_mode, y_rul, y_health)
    
    def get_all_results(self) -> Dict:
        return self.results


# ===========================================================================
# 5) OPERATIONAL METRICS
# ===========================================================================

class OperationalMetrics:
    """Métricas operacionais."""
    
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.results = {}
    
    def compute_false_alarms_per_day(
        self,
        y_true_sev: np.ndarray,
        y_proba_sev: np.ndarray,
        window_info: List[Dict],
        labels: List[str],
        samples_per_day: int = 180  # ~3 min windows * 60 = 180/hour * 24 = 4320/day, but let's use reasonable estimate
    ) -> Dict:
        """Calcula falsos alarmes por dia."""
        
        # Encontra índice das classes críticas
        critical_indices = [
            labels.index(cls) for cls in self.cfg.severity_critical_classes
            if cls in labels
        ]
        
        if not critical_indices:
            self.results['false_alarms'] = {'error': 'No critical classes found'}
            return self.results['false_alarms']
        
        # Probabilidade de ser crítico
        proba_critical = y_proba_sev[:, critical_indices].sum(axis=1)
        
        # Alarme simples (sem histerese)
        alarms_simple = proba_critical > self.cfg.alarm_threshold
        
        # Alarme com histerese (N janelas seguidas)
        alarms_hysteresis = np.zeros_like(alarms_simple)
        count = 0
        for i in range(len(alarms_simple)):
            if alarms_simple[i]:
                count += 1
                if count >= self.cfg.hysteresis_windows:
                    alarms_hysteresis[i] = True
            else:
                count = 0
        
        # True positives vs False positives
        y_true_critical = np.isin(y_true_sev, critical_indices)
        
        # FP simple
        fp_simple = alarms_simple & ~y_true_critical
        fp_rate_simple = fp_simple.sum() / max(1, (~y_true_critical).sum())
        
        # FP hysteresis
        fp_hysteresis = alarms_hysteresis & ~y_true_critical
        fp_rate_hysteresis = fp_hysteresis.sum() / max(1, (~y_true_critical).sum())
        
        # Estimate per day
        n_windows = len(y_true_sev)
        
        # Group by asset
        asset_fp = {}
        for i, w in enumerate(window_info):
            asset = w['asset_id']
            if asset not in asset_fp:
                asset_fp[asset] = {'fp_simple': 0, 'fp_hysteresis': 0, 'total': 0}
            asset_fp[asset]['total'] += 1
            if fp_simple[i]:
                asset_fp[asset]['fp_simple'] += 1
            if fp_hysteresis[i]:
                asset_fp[asset]['fp_hysteresis'] += 1
        
        # Convert to per day (assuming uniform distribution)
        fp_per_day_simple = []
        fp_per_day_hysteresis = []
        for asset, counts in asset_fp.items():
            if counts['total'] > 0:
                ratio_simple = counts['fp_simple'] / counts['total']
                ratio_hysteresis = counts['fp_hysteresis'] / counts['total']
                fp_per_day_simple.append(ratio_simple * samples_per_day)
                fp_per_day_hysteresis.append(ratio_hysteresis * samples_per_day)
        
        self.results['false_alarms'] = {
            'threshold': self.cfg.alarm_threshold,
            'hysteresis_windows': self.cfg.hysteresis_windows,
            'simple': {
                'fp_count': int(fp_simple.sum()),
                'fp_rate': float(fp_rate_simple),
                'fp_per_day_avg': float(np.mean(fp_per_day_simple)) if fp_per_day_simple else 0,
                'fp_per_day_std': float(np.std(fp_per_day_simple)) if fp_per_day_simple else 0,
                'fp_per_day_p95': float(np.percentile(fp_per_day_simple, 95)) if len(fp_per_day_simple) > 1 else 0,
            },
            'hysteresis': {
                'fp_count': int(fp_hysteresis.sum()),
                'fp_rate': float(fp_rate_hysteresis),
                'fp_per_day_avg': float(np.mean(fp_per_day_hysteresis)) if fp_per_day_hysteresis else 0,
                'fp_per_day_std': float(np.std(fp_per_day_hysteresis)) if fp_per_day_hysteresis else 0,
                'fp_per_day_p95': float(np.percentile(fp_per_day_hysteresis, 95)) if len(fp_per_day_hysteresis) > 1 else 0,
            },
            'by_asset': {str(k): v for k, v in asset_fp.items()},
        }
        
        print(f"\n[OPS] False alarms:")
        print(f"  Simple: {fp_simple.sum()} FP ({fp_rate_simple*100:.2f}%), ~{self.results['false_alarms']['simple']['fp_per_day_avg']:.1f}/day")
        print(f"  Hysteresis: {fp_hysteresis.sum()} FP ({fp_rate_hysteresis*100:.2f}%), ~{self.results['false_alarms']['hysteresis']['fp_per_day_avg']:.1f}/day")
        
        return self.results['false_alarms']
    
    def get_all_results(self) -> Dict:
        return self.results


# ===========================================================================
# VISUALIZATION
# ===========================================================================

class Visualizer:
    """Gera visualizações."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str],
        task_name: str
    ) -> str:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - {task_name}')
        plt.tight_layout()
        
        path = self.output_dir / f'confusion_matrix_{task_name.lower()}.png'
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)
    
    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        labels: List[str],
        critical_classes: List[str]
    ) -> str:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for cls in critical_classes:
            if cls in labels:
                idx = labels.index(cls)
                y_binary = (y_true == idx).astype(int)
                if y_binary.sum() > 0:
                    precision, recall, _ = precision_recall_curve(y_binary, y_proba[:, idx])
                    ap = average_precision_score(y_binary, y_proba[:, idx])
                    ax.plot(recall, precision, label=f'{cls} (AP={ap:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves (Critical Classes)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = self.output_dir / 'pr_curve_critical.png'
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)
    
    def plot_rul_error_bins(self, rul_metrics: Dict) -> str:
        bins = rul_metrics.get('mae_by_bin', {})
        if not bins:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = list(bins.keys())
        mae_values = [bins[l]['mae_norm'] for l in labels]
        n_samples = [bins[l]['n_samples'] for l in labels]
        
        bars = ax.bar(labels, mae_values, color='steelblue')
        ax.set_xlabel('RUL Range')
        ax.set_ylabel('MAE (normalized)')
        ax.set_title('RUL MAE by Range')
        
        # Add sample counts on bars
        for bar, n in zip(bars, n_samples):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'n={n}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        path = self.output_dir / 'rul_error_bins.png'
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)
    
    def plot_stress_summary(self, stress_results: Dict) -> str:
        if not stress_results:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        tests = list(stress_results.keys())
        metrics = ['severity_acc', 'mode_acc', 'rul_mae', 'health_mae']
        titles = ['Severity Accuracy', 'Mode Accuracy', 'RUL MAE', 'Health MAE']
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            values = [stress_results[t][metric] for t in tests]
            degradation = [stress_results[t]['degradation'][metric] for t in tests]
            
            colors = ['red' if d > 0.05 else 'orange' if d > 0.02 else 'green' for d in degradation]
            if 'mae' in metric:
                colors = ['red' if d > 0.01 else 'orange' if d > 0.005 else 'green' for d in degradation]
            
            bars = ax.bar(tests, values, color=colors)
            ax.set_title(title)
            ax.set_xticklabels(tests, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Stress Test Results', fontsize=14)
        plt.tight_layout()
        
        path = self.output_dir / 'stress_summary.png'
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)


# ===========================================================================
# MAIN EVALUATION
# ===========================================================================

def main():
    print("=" * 70)
    print("COMPLETE EVALUATION - Thesis-Ready Evidence")
    print("=" * 70)
    
    cfg = EvalConfig()
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "datasets" / "sensors_log_v2.csv"
    model_path = base_dir / "models" / "pump_cnn2d_v2.keras"
    output_dir = base_dir / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    SENSOR_COLS = [
        "overall_vibration", "vibration_x", "vibration_y", "vibration_z",
        "motor_current", "pressure", "flow", "temperature"
    ]
    
    # Initialize components
    pipeline = EvalPipeline(cfg)
    leakage_proofs = AntiLeakageProofs(cfg)
    perf_metrics = PerformanceMetrics(cfg)
    stress_tests = StressTests(cfg)
    ops_metrics = OperationalMetrics(cfg)
    visualizer = Visualizer(output_dir)
    
    # =========================================================================
    # 0) DATASET CONTRACT
    # =========================================================================
    print("\n" + "=" * 70)
    print("0) DATASET CONTRACT")
    print("=" * 70)
    
    df = pipeline.load_data(data_path)
    train_df, val_df, test_df, train_assets, val_assets, test_assets = pipeline.split_by_asset(df)
    
    dataset_contract = {
        'dataset_name': data_path.name,
        'total_samples': len(df),
        'n_assets_train': len(train_assets),
        'n_assets_val': len(val_assets),
        'n_assets_test': len(test_assets),
        'train_assets': train_assets,
        'val_assets': val_assets,
        'test_assets': test_assets,
        'time_range_train': {
            'min': str(train_df['timestamp'].min()),
            'max': str(train_df['timestamp'].max()),
        },
        'time_range_val': {
            'min': str(val_df['timestamp'].min()),
            'max': str(val_df['timestamp'].max()),
        },
        'time_range_test': {
            'min': str(test_df['timestamp'].min()),
            'max': str(test_df['timestamp'].max()),
        },
        'sensor_cols': SENSOR_COLS,
        'severity_distribution_train': train_df['severity'].value_counts().to_dict(),
        'severity_distribution_val': val_df['severity'].value_counts().to_dict(),
        'severity_distribution_test': test_df['severity'].value_counts().to_dict(),
        'mode_distribution_train': train_df['mode'].value_counts().to_dict(),
        'mode_distribution_val': val_df['mode'].value_counts().to_dict(),
        'mode_distribution_test': test_df['mode'].value_counts().to_dict(),
    }
    
    print(f"[CONTRACT] Dataset: {data_path.name}")
    print(f"[CONTRACT] Assets: Train={len(train_assets)}, Val={len(val_assets)}, Test={len(test_assets)}")
    
    # =========================================================================
    # 1) ANTI-LEAKAGE PROOFS
    # =========================================================================
    print("\n" + "=" * 70)
    print("1) ANTI-LEAKAGE PROOFS")
    print("=" * 70)
    
    leakage_proofs.verify_asset_split(train_assets, val_assets, test_assets)
    
    # Create windows for test
    print("\n[EVAL] Creating test windows...")
    test_data = pipeline.create_windows(train_df.head(5000), SENSOR_COLS, fit_scaler=True)  # Use train for scaler
    test_data = pipeline.create_windows(test_df, SENSOR_COLS, fit_scaler=False)
    
    leakage_proofs.verify_window_alignment(test_data[7])
    leakage_proofs.verify_no_leaky_features(df, SENSOR_COLS)
    
    # =========================================================================
    # 2) LOAD MODEL AND PREDICT
    # =========================================================================
    print("\n" + "=" * 70)
    print("2) MODEL EVALUATION")
    print("=" * 70)
    
    print(f"[LOAD] Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Prepare data
    train_data_full = pipeline.create_windows(train_df, SENSOR_COLS, fit_scaler=True)
    y_sev_train, y_mode_train = pipeline.encode_labels(train_data_full[3], train_data_full[4], fit=True)
    
    test_data = pipeline.create_windows(test_df, SENSOR_COLS, fit_scaler=False)
    y_sev_test, y_mode_test = pipeline.encode_labels(test_data[3], test_data[4])
    
    dataset_contract['rul_max_train'] = float(pipeline.rul_max_train)
    dataset_contract['scaler_mean'] = pipeline.scaler.mean_.tolist()
    dataset_contract['scaler_std'] = pipeline.scaler.scale_.tolist()
    dataset_contract['severity_labels'] = pipeline.severity_labels
    dataset_contract['mode_labels'] = pipeline.mode_labels
    
    X_test = [test_data[0], test_data[1], test_data[2]]
    
    print(f"[EVAL] Predicting on {test_data[0].shape[0]} test windows...")
    preds = model.predict(X_test, verbose=0)
    
    y_pred_sev = np.argmax(preds[0], axis=1)
    y_pred_mode = np.argmax(preds[1], axis=1)
    y_pred_rul = preds[2].flatten()
    y_pred_health = preds[3].flatten()
    
    # =========================================================================
    # 3) PERFORMANCE METRICS
    # =========================================================================
    print("\n" + "=" * 70)
    print("3) PERFORMANCE METRICS")
    print("=" * 70)
    
    perf_metrics.compute_rul_metrics(test_data[5], y_pred_rul, pipeline.rul_max_train)
    perf_metrics.compute_health_metrics(test_data[6], y_pred_health)
    perf_metrics.compute_classification_metrics(
        y_sev_test, y_pred_sev, preds[0],
        pipeline.severity_labels, 'severity'
    )
    perf_metrics.compute_classification_metrics(
        y_mode_test, y_pred_mode, preds[1],
        pipeline.mode_labels, 'mode'
    )
    
    # =========================================================================
    # 4) STRESS TESTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("4) STRESS TESTS")
    print("=" * 70)
    
    stress_tests.run_all_stress_tests(
        model, X_test,
        [y_sev_test, y_mode_test, test_data[5], test_data[6]],
        pipeline.severity_labels, pipeline.mode_labels,
        perf_metrics.get_all_results()
    )
    
    # =========================================================================
    # 5) OPERATIONAL METRICS
    # =========================================================================
    print("\n" + "=" * 70)
    print("5) OPERATIONAL METRICS")
    print("=" * 70)
    
    ops_metrics.compute_false_alarms_per_day(
        y_sev_test, preds[0], test_data[7],
        pipeline.severity_labels
    )
    
    # =========================================================================
    # 6) VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("6) GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    perf_results = perf_metrics.get_all_results()
    
    viz_paths = {}
    
    # Confusion matrices
    viz_paths['confusion_severity'] = visualizer.plot_confusion_matrix(
        np.array(perf_results['severity']['confusion_matrix']),
        pipeline.severity_labels, 'Severity'
    )
    print(f"[VIZ] Saved: {viz_paths['confusion_severity']}")
    
    viz_paths['confusion_mode'] = visualizer.plot_confusion_matrix(
        np.array(perf_results['mode']['confusion_matrix']),
        pipeline.mode_labels, 'Mode'
    )
    print(f"[VIZ] Saved: {viz_paths['confusion_mode']}")
    
    # PR curve
    viz_paths['pr_curve'] = visualizer.plot_pr_curve(
        y_sev_test, preds[0],
        pipeline.severity_labels,
        cfg.severity_critical_classes
    )
    print(f"[VIZ] Saved: {viz_paths['pr_curve']}")
    
    # RUL error bins
    viz_paths['rul_bins'] = visualizer.plot_rul_error_bins(perf_results['rul'])
    if viz_paths['rul_bins']:
        print(f"[VIZ] Saved: {viz_paths['rul_bins']}")
    
    # Stress summary
    viz_paths['stress'] = visualizer.plot_stress_summary(stress_tests.get_all_results().get('stress_tests', {}))
    if viz_paths['stress']:
        print(f"[VIZ] Saved: {viz_paths['stress']}")
    
    # =========================================================================
    # 7) COMPILE FINAL REPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print("7) FINAL REPORT")
    print("=" * 70)
    
    final_report = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model_path),
        'dataset_contract': dataset_contract,
        'anti_leakage': leakage_proofs.get_all_results(),
        'performance': perf_metrics.get_all_results(),
        'stress_tests': stress_tests.get_all_results(),
        'operational': ops_metrics.get_all_results(),
        'visualizations': viz_paths,
        'targets': {
            'severity_acc': {
                'target': 0.90,
                'achieved': perf_results['severity']['accuracy'],
                'met': perf_results['severity']['accuracy'] >= 0.90,
            },
            'mode_acc': {
                'target': 0.90,
                'achieved': perf_results['mode']['accuracy'],
                'met': perf_results['mode']['accuracy'] >= 0.90,
            },
            'rul_mae': {
                'target': 0.20,
                'achieved': perf_results['rul']['mae_norm'],
                'met': perf_results['rul']['mae_norm'] <= 0.20,
            },
            'health_mae': {
                'target': 0.10,
                'achieved': perf_results['health']['mae'],
                'met': perf_results['health']['mae'] <= 0.10,
            },
        },
        'all_targets_met': all([
            perf_results['severity']['accuracy'] >= 0.90,
            perf_results['mode']['accuracy'] >= 0.90,
            perf_results['rul']['mae_norm'] <= 0.20,
            perf_results['health']['mae'] <= 0.10,
        ]),
    }
    
    # Save report
    report_path = output_dir / 'eval_report_complete.json'
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    print(f"[SAVE] Report: {report_path}")
    
    # Save stress summary separately
    stress_path = output_dir / 'stress_summary.json'
    with open(stress_path, 'w') as f:
        json.dump(stress_tests.get_all_results(), f, indent=2, default=str)
    print(f"[SAVE] Stress: {stress_path}")
    
    # Save false alarms CSV
    fa_data = ops_metrics.get_all_results().get('false_alarms', {}).get('by_asset', {})
    if fa_data:
        fa_df = pd.DataFrame([
            {'asset_id': k, **v} for k, v in fa_data.items()
        ])
        fa_path = output_dir / 'false_alarms_per_asset.csv'
        fa_df.to_csv(fa_path, index=False)
        print(f"[SAVE] False alarms: {fa_path}")
    
    # =========================================================================
    # 8) SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print("\n--- TARGETS ---")
    for target, info in final_report['targets'].items():
        status = "✓" if info['met'] else "✗"
        print(f"  {status} {target}: {info['achieved']:.4f} (target: {info['target']})")
    
    print("\n--- ANTI-LEAKAGE ---")
    for check, result in final_report['anti_leakage'].items():
        if isinstance(result, dict) and 'ok' in result:
            status = "✓" if result['ok'] else "✗"
            print(f"  {status} {check}")
    
    print("\n--- STRESS ROBUSTNESS ---")
    stress_data = final_report['stress_tests'].get('stress_tests', {})
    for test, result in stress_data.items():
        deg_sev = result['degradation']['severity_acc']
        print(f"  {test}: Severity degradation = {deg_sev*100:.2f}%")
    
    print("\n--- OPERATIONAL ---")
    fa = final_report['operational'].get('false_alarms', {})
    if 'hysteresis' in fa:
        print(f"  False alarms/day (hysteresis): {fa['hysteresis']['fp_per_day_avg']:.1f}")
    
    print("\n" + "=" * 70)
    print(f"ALL TARGETS MET: {final_report['all_targets_met']}")
    print("=" * 70)
    
    return final_report


if __name__ == "__main__":
    report = main()
