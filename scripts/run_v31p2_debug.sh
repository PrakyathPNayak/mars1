#!/bin/bash
cd /workspace/mars1

# Trap signals
trap 'echo "CAUGHT SIGNAL: SIGHUP" >> logs/v31p2_signal.log' HUP
trap 'echo "CAUGHT SIGNAL: SIGTERM" >> logs/v31p2_signal.log' TERM
trap 'echo "CAUGHT SIGNAL: SIGINT" >> logs/v31p2_signal.log' INT
trap 'echo "CAUGHT SIGNAL: SIGPIPE" >> logs/v31p2_signal.log' PIPE

echo "Starting at $(date), PID=$$" >> logs/v31p2_signal.log

python3 -u src/training/train.py \
  --total-steps 3000000 \
  --n-envs 8 \
  --ckpt-dir checkpoints/v31p2 \
  --log-dir logs/v31p2_tb \
  >> logs/v31p2_train.log 2>&1

EXIT_CODE=$?
echo "Exited at $(date) with code=$EXIT_CODE" >> logs/v31p2_signal.log
