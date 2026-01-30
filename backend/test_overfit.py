#!/usr/bin/env python3
"""
OVERFIT TEST: Can the model memorize 256 samples?
If train acc stays ~0.20, then labels are random/broken.
If train acc -> 0.95+, then labels are learnable (problem is elsewhere).
"""

import sys
import subprocess

# Run training with overfit config
print("=" * 80)
print("OVERFIT TEST: Training on 256 samples for 50 epochs")
print("Config: dropout=0, l2=0, sensor_dropout=0, augment=OFF, lr=1e-3")
print("=" * 80)

cmd = [
    "python3",
    "train_pump_predictive_market.py",
    "--epochs", "50",
    "--overfit_test", "256"  # Will add this flag to main script
]

result = subprocess.run(cmd, cwd="/mnt/c/Users/msmig/Desktop/Tese/DigitalTwin/backend")
sys.exit(result.returncode)
