#!/bin/bash

# Activate environment
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export MUJOCO_GL=egl

MODEL="checkpoints/best/best_model.zip"

echo "================ WALKING (FLAT) ================"
python3 terrain_testing/scripts/run_single_terrain.py --terrain flat --checkpoint $MODEL --steps 200 --render human

echo "================ SLIDING (ICE) ================"
python3 terrain_testing/scripts/run_single_terrain.py --terrain frozen_lake --checkpoint $MODEL --steps 200 --render human

echo "================ STAIRS ========================"
python3 terrain_testing/scripts/run_single_terrain.py --terrain pyramid_stairs --checkpoint $MODEL --steps 200 --render human

echo "================ JUMP (HOLE) ==================="
python3 terrain_testing/scripts/run_single_terrain.py --terrain trench_crossing --checkpoint $MODEL --steps 200 --render human

echo "================ BALANCE (ROUGH) ==============="
python3 terrain_testing/scripts/run_single_terrain.py --terrain rubble_field --checkpoint $MODEL --steps 200 --render none
