#!/usr/bin/env python3
"""
Modelo Híbrido Final para Pump Predictive Maintenance
- HistGradientBoosting para Severity (mais potente que RF)
- RandomForest para Mode
- Ridge Regression para RUL/Health
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.linear_model import Ridge
import joblib
import json
from pathlib import Path
import time

def main():
    start_time = time.time()
    
    print("=" * 70)
    print("MODELO HÍBRIDO FINAL - PUMP PREDICTIVE MAINTENANCE")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv('logs/sensors_log.csv', low_memory=False)
    print(f"\n[INFO] Dataset: {len(df)} linhas")
    
    # Features
    sensor_cols = ['overall_vibration', 'vibration_x', 'vibration_y', 'vibration_z', 
                   'motor_current', 'pressure', 'flow', 'temperature']
    sensor_cols = [c for c in sensor_cols if c in df.columns]
    print(f"[INFO] Usando {len(sensor_cols)} features: {sensor_cols}")
    
    X = df[sensor_cols].fillna(0).values.astype(np.float32)
    y_mode = df['mode'].values
    y_sev = df['severity'].values
    y_rul = df['rul_minutes'].fillna(500).values / 1000.0  # normalize 0-1
    y_health = df['health_index'].fillna(50).values / 100.0  # normalize 0-1
    
    # Split temporal 70/15/15
    n = len(X)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    X_train = X[:n_train]
    X_val = X[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    
    y_sev_tr, y_sev_va, y_sev_te = y_sev[:n_train], y_sev[n_train:n_train+n_val], y_sev[n_train+n_val:]
    y_mode_tr, y_mode_va, y_mode_te = y_mode[:n_train], y_mode[n_train:n_train+n_val], y_mode[n_train+n_val:]
    y_rul_tr, y_rul_va, y_rul_te = y_rul[:n_train], y_rul[n_train:n_train+n_val], y_rul[n_train+n_val:]
    y_health_tr, y_health_va, y_health_te = y_health[:n_train], y_health[n_train:n_train+n_val], y_health[n_train+n_val:]
    
    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Normalize
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_val_n = scaler.transform(X_val)
    X_test_n = scaler.transform(X_test)
    
    # ========== TRAIN SEVERITY ==========
    print("\n[TRAINING] HistGradientBoostingClassifier para SEVERITY...")
    print("           (max_iter=500, max_depth=10, learning_rate=0.05)")
    hgb_sev = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=10,
        learning_rate=0.05,
        l2_regularization=0.1,
        random_state=42
    )
    hgb_sev.fit(X_train_n, y_sev_tr)
    sev_val = accuracy_score(y_sev_va, hgb_sev.predict(X_val_n))
    sev_test = accuracy_score(y_sev_te, hgb_sev.predict(X_test_n))
    print(f"           Val acc: {sev_val:.4f}")
    print(f"           Test acc: {sev_test:.4f}")
    
    # ========== TRAIN MODE ==========
    print("\n[TRAINING] RandomForestClassifier para MODE...")
    print("           (n_estimators=400, max_depth=30)")
    rf_mode = RandomForestClassifier(
        n_estimators=400,
        max_depth=30,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_mode.fit(X_train_n, y_mode_tr)
    mode_val = accuracy_score(y_mode_va, rf_mode.predict(X_val_n))
    mode_test = accuracy_score(y_mode_te, rf_mode.predict(X_test_n))
    print(f"           Val acc: {mode_val:.4f}")
    print(f"           Test acc: {mode_test:.4f}")
    
    # ========== TRAIN RUL ==========
    print("\n[TRAINING] Ridge Regression para RUL...")
    ridge_rul = Ridge(alpha=1.0)
    ridge_rul.fit(X_train_n, y_rul_tr)
    rul_val = mean_absolute_error(y_rul_va, np.clip(ridge_rul.predict(X_val_n), 0, 1))
    rul_test = mean_absolute_error(y_rul_te, np.clip(ridge_rul.predict(X_test_n), 0, 1))
    print(f"           Val MAE: {rul_val:.4f}")
    print(f"           Test MAE: {rul_test:.4f}")
    
    # ========== TRAIN HEALTH ==========
    print("\n[TRAINING] Ridge Regression para HEALTH...")
    ridge_health = Ridge(alpha=1.0)
    ridge_health.fit(X_train_n, y_health_tr)
    health_val = mean_absolute_error(y_health_va, np.clip(ridge_health.predict(X_val_n), 0, 1))
    health_test = mean_absolute_error(y_health_te, np.clip(ridge_health.predict(X_test_n), 0, 1))
    print(f"           Val MAE: {health_val:.4f}")
    print(f"           Test MAE: {health_test:.4f}")
    
    # ========== FINAL RESULTS ==========
    elapsed = time.time() - start_time
    
    print("\n")
    print("=" * 70)
    print("                       FINAL TEST RESULTS")
    print("=" * 70)
    print(f"  RUL MAE:      {rul_test:.4f}   (target < 0.20)  {'✓ PASS' if rul_test < 0.2 else '✗ FAIL'}")
    print(f"  Health MAE:   {health_test*100:.2f}%   (target < 10%)   {'✓ PASS' if health_test < 0.1 else '✗ FAIL'}")
    print(f"  Severity acc: {sev_test:.4f}   (target > 0.90)  {'✓ PASS' if sev_test > 0.9 else '✗ FAIL'}")
    print(f"  Mode acc:     {mode_test:.4f}   (target > 0.90)  {'✓ PASS' if mode_test > 0.9 else '✗ FAIL'}")
    print("=" * 70)
    
    targets_met = sum([
        rul_test < 0.2,
        health_test < 0.1,
        sev_test > 0.9,
        mode_test > 0.9
    ])
    print(f"  TARGETS MET: {targets_met}/4")
    print(f"  TRAINING TIME: {elapsed:.1f}s")
    print("=" * 70)
    
    # ========== SAVE MODELS ==========
    out_dir = Path('models')
    out_dir.mkdir(exist_ok=True)
    
    joblib.dump(hgb_sev, out_dir / 'severity_hgb.joblib')
    joblib.dump(rf_mode, out_dir / 'mode_rf.joblib')
    joblib.dump(ridge_rul, out_dir / 'rul_ridge.joblib')
    joblib.dump(ridge_health, out_dir / 'health_ridge.joblib')
    joblib.dump(scaler, out_dir / 'scaler_hybrid.joblib')
    
    # Save report
    report = {
        "test_results": {
            "rul_mae": float(rul_test),
            "health_mae_percent": float(health_test * 100),
            "severity_acc": float(sev_test),
            "mode_acc": float(mode_test),
        },
        "val_results": {
            "rul_mae": float(rul_val),
            "health_mae_percent": float(health_val * 100),
            "severity_acc": float(sev_val),
            "mode_acc": float(mode_val),
        },
        "targets_met": {
            "rul": bool(rul_test < 0.2),
            "health": bool(health_test < 0.1),
            "severity": bool(sev_test > 0.9),
            "mode": bool(mode_test > 0.9),
        },
        "models_used": {
            "severity": "HistGradientBoostingClassifier",
            "mode": "RandomForestClassifier",
            "rul": "Ridge Regression",
            "health": "Ridge Regression"
        },
        "training_time_seconds": elapsed
    }
    
    with open(out_dir / 'eval_report_hybrid.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[SAVED] Models to {out_dir}/")
    
    # ========== CLASSIFICATION REPORTS ==========
    print("\n")
    print("=" * 70)
    print("SEVERITY CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_sev_te, hgb_sev.predict(X_test_n), zero_division=0))
    
    print("=" * 70)
    print("MODE CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_mode_te, rf_mode.predict(X_test_n), zero_division=0))
    
    return report

if __name__ == '__main__':
    main()
