#!/bin/bash
# Auto-restart training loop for v31p
# Resumes from latest checkpoint on death

# CRITICAL: Limit threads to prevent PyTorch/MKL deadlock
# Without this, training freezes at ~1-3 iterations (59 threads, 1840% CPU, zero progress)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

CKPT_DIR="checkpoints/v31p"
LOG_DIR="logs/training"
TOTAL_STEPS=5000000
N_ENVS=8

find_latest_ckpt() {
    ls -t "$CKPT_DIR"/cheetah_ppo_*_steps.zip 2>/dev/null | head -1
}

ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))
    LATEST=$(find_latest_ckpt)
    
    if [ -z "$LATEST" ]; then
        echo "[attempt $ATTEMPT] No checkpoint found. Starting from v31o/1M..."
        RESUME="checkpoints/v31o/cheetah_ppo_1000000_steps.zip"
    else
        echo "[attempt $ATTEMPT] Resuming from: $LATEST"
        RESUME="$LATEST"
    fi
    
    python3 -u src/training/train.py \
        --total-steps $TOTAL_STEPS \
        --n-envs $N_ENVS \
        --ckpt-dir "$CKPT_DIR" \
        --log-dir "$LOG_DIR" \
        --resume "$RESUME"
    
    EXIT_CODE=$?
    echo "[attempt $ATTEMPT] Training exited with code $EXIT_CODE at $(date)"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed normally. Done."
        break
    fi
    
    echo "Restarting in 10 seconds..."
    sleep 10
done
