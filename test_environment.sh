#!/bin/bash

# Activate environment
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export MUJOCO_GL=egl

MODEL="runs/best_model.zip"

echo "================ WALKING (FLAT) ================"
python terrain_testing/scripts/run_single_terrain.py --terrain flat --checkpoint $MODEL --steps 200 --render none

echo "================ SLIDING (ICE) ================"
python terrain_testing/scripts/run_single_terrain.py --terrain frozen_lake --checkpoint $MODEL --steps 200 --render none

echo "================ STAIRS ========================"
python terrain_testing/scripts/run_single_terrain.py --terrain pyramid_stairs --checkpoint $MODEL --steps 200 --render none

echo "================ JUMP (HOLE) ==================="
python terrain_testing/scripts/run_single_terrain.py --terrain trench_crossing --checkpoint $MODEL --steps 200 --render none

echo "================ BALANCE (ROUGH) ==============="
python terrain_testing/scripts/run_single_terrain.py --terrain rubble_field --checkpoint $MODEL --steps 200 --render none
