#!/usr/bin/env python3
"""
Fast Training Script for Pump Predictive Maintenance
Uses optimized models with stratified split
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.linear_model import Ridge
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 70)
    print("FAST TRAINING - STRATIFIED SPLIT")
    print("=" * 70)
    
    df = pd.read_csv('logs/sensors_log.csv', low_memory=False)
    print(f"\n[INFO] Dataset: {len(df)} linhas")
    
    sensor_cols = ['overall_vibration', 'vibration_x', 'vibration_y', 'vibration_z', 
                   'motor_current', 'pressure', 'flow', 'temperature']
    sensor_cols = [c for c in sensor_cols if c in df.columns]
    
    X = df[sensor_cols].fillna(0).values.astype(np.float32)
    y_sev = df['severity'].values
    y_mode = df['mode'].values
    y_rul = df['rul_minutes'].fillna(500).values / 1000.0
    y_health = df['health_index'].fillna(50).values / 100.0
    
    # Stratified split using severity (main concern)
    X_train, X_test, y_sev_tr, y_sev_te, y_mode_tr, y_mode_te, y_rul_tr, y_rul_te, y_health_tr, y_health_te = train_test_split(
        X, y_sev, y_mode, y_rul, y_health,
        test_size=0.20,
        stratify=y_sev,
        random_state=42
    )
    
    print(f"[INFO] Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Normalize
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_test_n = scaler.transform(X_test)
    
    # SEVERITY - RandomForest (fast)
    print("\n[TRAINING] RandomForest para SEVERITY...")
    rf_sev = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_sev.fit(X_train_n, y_sev_tr)
    sev_test = accuracy_score(y_sev_te, rf_sev.predict(X_test_n))
    print(f"           Test acc: {sev_test:.4f}")
    
    # MODE - RandomForest
    print("\n[TRAINING] RandomForest para MODE...")
    rf_mode = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_mode.fit(X_train_n, y_mode_tr)
    mode_test = accuracy_score(y_mode_te, rf_mode.predict(X_test_n))
    print(f"           Test acc: {mode_test:.4f}")
    
    # RUL & Health
    ridge_rul = Ridge(alpha=1.0).fit(X_train_n, y_rul_tr)
    ridge_health = Ridge(alpha=1.0).fit(X_train_n, y_health_tr)
    rul_test = mean_absolute_error(y_rul_te, np.clip(ridge_rul.predict(X_test_n), 0, 1))
    health_test = mean_absolute_error(y_health_te, np.clip(ridge_health.predict(X_test_n), 0, 1))
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  RUL MAE:      {rul_test:.4f}   (target < 0.20)  {'✓' if rul_test < 0.2 else '✗'}")
    print(f"  Health MAE:   {health_test*100:.2f}%   (target < 10%)   {'✓' if health_test < 0.1 else '✗'}")
    print(f"  Severity acc: {sev_test:.4f}   (target > 0.90)  {'✓' if sev_test > 0.9 else '✗'}")
    print(f"  Mode acc:     {mode_test:.4f}   (target > 0.90)  {'✓' if mode_test > 0.9 else '✗'}")
    print("=" * 70)
    
    targets = sum([rul_test < 0.2, health_test < 0.1, sev_test > 0.9, mode_test > 0.9])
    print(f"\nTARGETS MET: {targets}/4")
    
    # Save models
    out_dir = Path('models')
    out_dir.mkdir(exist_ok=True)
    joblib.dump(rf_sev, out_dir / 'severity_rf.joblib')
    joblib.dump(rf_mode, out_dir / 'mode_rf.joblib')
    joblib.dump(ridge_rul, out_dir / 'rul_ridge.joblib')
    joblib.dump(ridge_health, out_dir / 'health_ridge.joblib')
    joblib.dump(scaler, out_dir / 'scaler.joblib')
    
    # Save report
    report = {
        "test_results": {
            "rul_mae": float(rul_test),
            "health_mae_percent": float(health_test * 100),
            "severity_acc": float(sev_test),
            "mode_acc": float(mode_test),
        },
        "targets_met": {
            "rul": bool(rul_test < 0.2),
            "health": bool(health_test < 0.1),
            "severity": bool(sev_test > 0.9),
            "mode": bool(mode_test > 0.9),
        }
    }
    with open(out_dir / 'eval_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[SAVED] Models to {out_dir}/")
    
    print("\n=== SEVERITY REPORT ===")
    print(classification_report(y_sev_te, rf_sev.predict(X_test_n), zero_division=0))
    
    print("\n=== MODE REPORT ===")
    print(classification_report(y_mode_te, rf_mode.predict(X_test_n), zero_division=0))

if __name__ == '__main__':
    main()
