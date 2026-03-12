#!/usr/bin/env bash
# Watch current L2 training, run eval when done, then start v2 training.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

TRAIN_PID=1817
LOG=/tmp/l2_training.log
V2_LOG=/tmp/l2_training_v2.log

echo "[$(date +%H:%M:%S)] Waiting for PID $TRAIN_PID to finish..."

# Wait for current training to complete
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
done

echo "[$(date +%H:%M:%S)] Training PID $TRAIN_PID finished."
echo ""
echo "=== Final training log ==="
tail -10 "$LOG"
echo ""

# Step 1: Eval current model (baseline)
echo "[$(date +%H:%M:%S)] === Running L2 eval (baseline) ==="
PYTHONUNBUFFERED=1 python scripts/eval_l2_model.py --model models/level2/final.pt --device cuda \
    2>&1 | tee /tmp/l2_eval_baseline.log

echo ""
echo "[$(date +%H:%M:%S)] === Starting v2 training (no augment, early stopping, multi-task) ==="

# Step 2: Retrain with v2 improvements
PYTHONUNBUFFERED=1 python scripts/train_l2_gnn_large.py \
    --skip-download \
    --skip-labels \
    --zinc-count 20000 \
    --augment 1 \
    --batch-size 128 \
    --epochs 300 \
    --finetune-epochs 60 \
    --warmup 20 \
    --patience 50 \
    --param-loss-weight 0.3 \
    2>&1 | tee "$V2_LOG"

echo ""
echo "[$(date +%H:%M:%S)] === Running L2 eval (v2) ==="

# Step 3: Eval v2 model
PYTHONUNBUFFERED=1 python scripts/eval_l2_model.py --model models/level2/final.pt --device cuda \
    2>&1 | tee /tmp/l2_eval_v2.log

echo ""
echo "[$(date +%H:%M:%S)] === All done ==="
echo "Baseline eval: /tmp/l2_eval_baseline.log"
echo "V2 training:   $V2_LOG"
echo "V2 eval:       /tmp/l2_eval_v2.log"
