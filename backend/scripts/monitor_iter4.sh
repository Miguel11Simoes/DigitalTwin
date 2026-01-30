#!/bin/bash
# Monitor ITERATION 4 training and compare to targets

LOGFILE="/tmp/train_15ep_iter4.log"
TARGETS_SEV=0.85
TARGETS_MODE=0.80
TARGETS_TOP2=0.92

echo "=== ITERATION 4 MONITORING ==="
echo "Waiting for training to complete..."

# Wait for training to finish
while ps aux | grep -v grep | grep -q "train_pump_predictive_market.py --epochs 15"; do
    sleep 10
done

echo "Training completed! Analyzing results..."

# Extract final metrics
FINAL_EPOCH=$(grep "VAL PRODUCT" "$LOGFILE" | tail -1)

if [ -z "$FINAL_EPOCH" ]; then
    echo "ERROR: No results found"
    exit 1
fi

# Extract metrics using grep and awk
FULL_LOG=$(tail -200 "$LOGFILE")
SEV_ACC=$(echo "$FULL_LOG" | grep -oP "val_severity_acc: \K[0-9.]+" | tail -1)
MODE_ACC=$(echo "$FULL_LOG" | grep -oP "val_mode_acc: \K[0-9.]+" | tail -1)
MODE_TOP2=$(echo "$FULL_LOG" | grep -oP "val_mode_top2: \K[0-9.]+" | tail -1)

echo "=== FINAL RESULTS ==="
echo "val_severity_acc: $SEV_ACC (target: $TARGETS_SEV)"
echo "val_mode_acc: $MODE_ACC (target: $TARGETS_MODE)"
echo "val_mode_top2: $MODE_TOP2 (target: $TARGETS_TOP2)"

# Check if targets met
SEV_MET=$(awk -v a="$SEV_ACC" -v t="$TARGETS_SEV" 'BEGIN{print (a>=t?"YES":"NO")}')
MODE_MET=$(awk -v a="$MODE_ACC" -v t="$TARGETS_MODE" 'BEGIN{print (a>=t?"YES":"NO")}')
TOP2_MET=$(awk -v a="$MODE_TOP2" -v t="$TARGETS_TOP2" 'BEGIN{print (a>=t?"YES":"NO")}')

echo ""
echo "=== TARGETS STATUS ==="
echo "Severity: $SEV_MET ($SEV_ACC vs $TARGETS_SEV)"
echo "Mode: $MODE_MET ($MODE_ACC vs $TARGETS_MODE)"
echo "Top2: $TOP2_MET ($MODE_TOP2 vs $TARGETS_TOP2)"

if [ "$SEV_MET" = "YES" ] && [ "$MODE_MET" = "YES" ] && [ "$TOP2_MET" = "YES" ]; then
    echo ""
    echo "✅ ALL TARGETS MET! Ready for full training (120 epochs)"
else
    echo ""
    echo "❌ TARGETS NOT MET - Need ITERATION 5"
    echo "Suggested improvements:"
    echo "  - Increase loss_weights further (sev 15→20, mode 10→15)"
    echo "  - Add more regularization (L2 1e-4→1e-3)"
    echo "  - Try label smoothing reduction"
fi
