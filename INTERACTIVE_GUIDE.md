# Interactive Controls Guide

This guide shows how to run the Unitree Go1 robot with keyboard controls in various terrain environments.

## Quick Start

### Option 1: Using `run.py` (Recommended)

```bash
# Default: mixed terrain with moderate difficulty
python3 run.py demo

# Specific terrain: flat, rough, slope_up, slope_down, stairs_up, stairs_down, gaps, stepping_stones, random_blocks
python3 run.py demo --terrain stairs_up --terrain-difficulty 0.5

# With terminal input (no need to click viewer window)
python3 run.py demo --terrain-input

# Run without trained policy (PD controller only)
python3 run.py demo --no-policy
```

### Option 2: Direct Script

```bash
# Uses mixed terrain (stairs, slopes, rough, gaps) by default
python3 scripts/interactive_control.py

# With a specific checkpoint
python3 scripts/interactive_control.py --checkpoint checkpoints/best/best_model.zip
```

## Keyboard Controls

```
FORWARD / BACKWARD / STRAFE:
  W or ↑     → Forward
  S or ↓     → Backward
  A or ←     → Strafe left
  D or →     → Strafe right

ROTATION:
  Q          → Turn left
  E          → Turn right

GAITS & MODES:
  1          → Walk (slow)
  2          → Trot (medium)
  3          → Run (fast)
  J          → Jump
  C          → Crouch

CONTROL:
  Space      → Stop all motion
  ESC        → Exit
```

## Terrain Types

The interactive demo now features **mixed terrain** by default (difficulty 0.6), which includes:

- **Stairs**: Both upward and downward climbing challenges
- **Slopes**: Ascending and descending inclines
- **Rough terrain**: Bumpy, uneven surfaces
- **Stepping stones**: Discrete footholds with gaps
- **Random blocks**: Obstacle field

### Terrain Difficulty Levels

```
0.0 - 0.2   → Easy (small obstacles, gentle slopes)
0.2 - 0.4   → Moderate (mixed terrain, 15-20° slopes)
0.4 - 0.7   → Challenging (steep obstacles, tight gaps)
0.7 - 1.0   → Expert (severe terrain, large elevation changes)
```

## Available Terrain Types

| Type | Description |
|------|-------------|
| `flat` | Completely flat ground (baseline) |
| `rough` | Bumpy, uneven surface |
| `slope_up` | Upward incline |
| `slope_down` | Downward incline |
| `stairs_up` | Staircase climbing |
| `stairs_down` | Staircase descent |
| `gaps` | Stepping stones with gaps |
| `stepping_stones` | Discrete footholds |
| `random_blocks` | Random obstacle field |
| `mixed` | Combination of all terrain types |

## Example Usage

```bash
# Explore on challenging stairs
python3 run.py demo --terrain stairs_up --terrain-difficulty 0.7

# Test on rough ground at difficulty 0.4
python3 run.py demo --terrain rough --terrain-difficulty 0.4

# Full mixed terrain (the default)
python3 run.py demo

# Without policy (just PD controller for testing)
python3 run.py demo --no-policy --terrain mixed --terrain-difficulty 0.3
```

## Features

✓ **Live visualization** with MuJoCo viewer
✓ **Keyboard-driven control** with terminal or pynput backend
✓ **Diverse terrain** including stairs, slopes, rough patches, and gaps
✓ **Multiple gaits**: walk, trot, run
✓ **Dynamic behaviors**: jump, crouch, turn
✓ **Trained policy** for intelligent locomotion (or PD-only fallback)
✓ **Real-time rendering** at simulation speed

## Requirements

- Python 3.8+
- mujoco >= 2.3.0
- gymnasium >= 0.26
- stable-baselines3
- pynput (optional, for mouse/keyboard input)

## Implementation Details

The interactive controls system uses:

1. **MiniCheetahEnv** (terrain-enabled): Provides the 196-dim observation and 12-dim action space
2. **Trained SB3 policy**: Takes observations → predicts actions for intelligent locomotion
3. **Keyboard controller**: Maps keyboard input to velocity/angular commands
4. **TerrainGenerator**: Creates diverse, challenging terrain dynamically

### Code References

- Main interactive script: `scripts/interactive_control.py`
- Viewer wrapper: `src/visualization/viewer.py`
- Keyboard controller: `src/control/keyboard_controller.py`
- Environment: `src/env/cheetah_env.py`
- Terrain generator: `src/terrain/terrain_gen.py`

