#!/bin/bash
# Auto-restart training loop for v31p2 with thread limits
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /workspace/mars1
CKPT_DIR="checkpoints/v31p2"
TOTAL=3000000
ENVS=8
ATTEMPT=0

find_latest() { ls -t "$CKPT_DIR"/cheetah_ppo_*_steps.zip 2>/dev/null | head -1; }

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    LATEST=$(find_latest)
    if [ -z "$LATEST" ]; then
        echo "[attempt $ATTEMPT] Fresh start..."
        python3 -u src/training/train.py --total-steps $TOTAL --n-envs $ENVS \
            --ckpt-dir "$CKPT_DIR" --log-dir logs/v31p2_tb \
            >> logs/v31p2_train.log 2>&1
    else
        echo "[attempt $ATTEMPT] Resume from: $LATEST"
        python3 -u src/training/train.py --total-steps $TOTAL --n-envs $ENVS \
            --ckpt-dir "$CKPT_DIR" --log-dir logs/v31p2_tb \
            --resume "$LATEST" \
            >> logs/v31p2_train.log 2>&1
    fi
    EC=$?
    echo "[attempt $ATTEMPT] Exit code=$EC at $(date)" >> logs/v31p2_train.log
    [ $EC -eq 0 ] && break
    sleep 5
done
