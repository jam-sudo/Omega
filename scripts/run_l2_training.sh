#!/bin/bash
# Auto-restart L2 training with --resume on crash
# Runs in a loop: if training exits non-zero, restart with --resume
# Only stops on: success (exit 0) or 3 consecutive failures

cd /home/jam/Omega
source .venv/bin/activate
export PYTHONUNBUFFERED=1

LOG=/tmp/l2_training_v5.log
MAX_RETRIES=3
RETRIES=0

# v5: fixed labels (diverse logP, clint_L_h, peff) + new surrogate
ARGS="--skip-download --skip-labels --augment 1 --patience 50 --param-loss-weight 0.3 --epochs 300 --finetune-epochs 60"

while [ $RETRIES -lt $MAX_RETRIES ]; do
    if [ $RETRIES -eq 0 ] && [ ! -f models/level2/stage_checkpoint.pt ]; then
        echo "$(date) [launcher] Starting fresh training (v5)..." | tee -a "$LOG"
        python scripts/train_l2_gnn_large.py $ARGS 2>&1 | tee -a "$LOG"
    else
        echo "$(date) [launcher] Resuming from checkpoint (retry $RETRIES)..." | tee -a "$LOG"
        python scripts/train_l2_gnn_large.py $ARGS --resume 2>&1 | tee -a "$LOG"
    fi

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date) [launcher] Training completed successfully!" | tee -a "$LOG"
        exit 0
    fi

    RETRIES=$((RETRIES + 1))
    echo "$(date) [launcher] Training crashed (exit $EXIT_CODE). Retry $RETRIES/$MAX_RETRIES in 5s..." | tee -a "$LOG"
    sleep 5
done

echo "$(date) [launcher] Max retries reached. Check logs at $LOG" | tee -a "$LOG"
exit 1
