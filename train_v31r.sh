#!/bin/bash
set -e
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

CKPT_DIR="checkpoints/v31r"
LOG_DIR="logs/v31r_tb"
TOTAL=5000000

while true; do
    LATEST=$(ls -t ${CKPT_DIR}/cheetah_ppo_*_steps.zip 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "[$(date)] Resuming from $LATEST"
        python3 -u src/training/train.py \
            --total-steps $TOTAL --n-envs 8 \
            --ckpt-dir $CKPT_DIR --log-dir $LOG_DIR \
            --resume "$LATEST"
    else
        echo "[$(date)] Starting fresh"
        python3 -u src/training/train.py \
            --total-steps $TOTAL --n-envs 8 \
            --ckpt-dir $CKPT_DIR --log-dir $LOG_DIR
    fi
    EXIT=$?
    echo "[$(date)] Training exited with code $EXIT"
    if [ $EXIT -eq 0 ]; then
        echo "Training completed successfully"
        break
    fi
    echo "Restarting in 5s..."
    sleep 5
done
