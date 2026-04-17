#!/bin/bash
# CRITICAL: Limit threads to prevent PyTorch/MKL deadlock
# Without this, training freezes at ~1-3 iterations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /workspace/mars1
exec python3 -u src/training/train.py \
  --total-steps 3000000 \
  --n-envs 8 \
  --ckpt-dir checkpoints/v31p2 \
  --log-dir logs/v31p2_tb \
  >> logs/v31p2_train.log 2>&1
