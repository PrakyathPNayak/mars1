"""
Visualize all registered terrain heightfields using matplotlib.

No MuJoCo installation required — pure NumPy / matplotlib.

Usage:
    # From repo root:
    python terrain_testing/scripts/visualize_heightfields.py

    # Show only paper terrains:
    python terrain_testing/scripts/visualize_heightfields.py --source paper

    # Show a single terrain at multiple difficulties:
    python terrain_testing/scripts/visualize_heightfields.py --terrain rubble_field --difficulties 0.1 0.5 1.0

    # Save to PNG instead of showing:
    python terrain_testing/scripts/visualize_heightfields.py --save
"""

import argparse
import math
import os
import sys

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_TERRAIN_TESTING = os.path.dirname(_HERE)
if _TERRAIN_TESTING not in sys.path:
    sys.path.insert(0, _TERRAIN_TESTING)

from maps.map_registry import REGISTRY, list_terrains, generate


def visualize_all(source=None, save=False, out_dir=None, resolution=100, seed=42):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    names = list_terrains(source=source)
    n = len(names)
    cols = 5
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = np.array(axes).flatten()

    for idx, name in enumerate(names):
        heights = generate(name, resolution=resolution, difficulty=0.5, seed=seed)
        entry = REGISTRY[name]
        ax = axes[idx]
        im = ax.imshow(heights, cmap="terrain", origin="lower",
                       aspect="equal",
                       vmin=min(-0.05, heights.min()),
                       vmax=max(0.25, heights.max()))
        ax.set_title(f"{name}\n[{entry.source}]", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.04, label="m")

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Terrain Heightfields (difficulty=0.5)", fontsize=14, y=1.01)
    plt.tight_layout()

    if save:
        out_path = os.path.join(out_dir or _TERRAIN_TESTING, "results", "heightfields.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()


def visualize_one(terrain_name, difficulties, save=False, out_dir=None,
                  resolution=150, seed=42):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    n = len(difficulties)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    if n == 1:
        axes = [axes]

    for ax, d in zip(axes, difficulties):
        heights = generate(terrain_name, resolution=resolution, difficulty=d, seed=seed)
        im = ax.imshow(heights, cmap="terrain", origin="lower", aspect="equal")
        ax.set_title(f"{terrain_name}\ndifficulty={d:.1f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.05, label="m")

    plt.tight_layout()

    if save:
        safe_name = terrain_name.replace("/", "_")
        out_path = os.path.join(out_dir or _TERRAIN_TESTING, "results",
                                f"{safe_name}_difficulties.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Preview terrain heightfields")
    parser.add_argument("--terrain", default=None,
                        help="Single terrain name (omit to show all)")
    parser.add_argument("--source", default=None, choices=["paper", "custom"],
                        help="Filter by source")
    parser.add_argument("--difficulties", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0],
                        help="Difficulty levels to show for --terrain mode")
    parser.add_argument("--resolution", type=int, default=100,
                        help="Heightfield resolution (default 100 for speed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true",
                        help="Save PNG instead of showing interactively")
    args = parser.parse_args()

    if args.terrain:
        visualize_one(args.terrain, args.difficulties, save=args.save,
                      resolution=args.resolution, seed=args.seed)
    else:
        visualize_all(source=args.source, save=args.save,
                      resolution=args.resolution, seed=args.seed)


if __name__ == "__main__":
    main()
