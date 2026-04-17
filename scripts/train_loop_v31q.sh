#!/bin/bash
# v31q training loop — auto-restart on crash
# Walk base_amp=0.0 (no reference free lunch) + moderate overshoot penalties

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

CKPT_DIR="checkpoints/v31q"
LOG_DIR="logs/v31q_tb"
TOTAL_STEPS=5000000

cd /workspace/mars1

while true; do
    # Find latest checkpoint to resume from
    LATEST=$(ls -t ${CKPT_DIR}/cheetah_ppo_*_steps.zip 2>/dev/null | head -1)
    
    if [ -n "$LATEST" ]; then
        echo "=== Resuming from $LATEST ==="
        python3 -u src/training/train.py \
            --total-steps $TOTAL_STEPS \
            --n-envs 8 \
            --ckpt-dir $CKPT_DIR \
            --log-dir $LOG_DIR \
            --resume "$LATEST" 2>&1 | tail -5
    else
        echo "=== Starting fresh v31q training ==="
        python3 -u src/training/train.py \
            --total-steps $TOTAL_STEPS \
            --n-envs 8 \
            --ckpt-dir $CKPT_DIR \
            --log-dir $LOG_DIR 2>&1 | tail -5
    fi
    
    EXIT_CODE=$?
    echo "=== Training exited with code $EXIT_CODE at $(date) ==="
    echo "Restarting in 5 seconds..."
    sleep 5
done
