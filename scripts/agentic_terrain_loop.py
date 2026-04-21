#!/usr/bin/env python3
"""
Agentic terrain-repair loop for Unitree Go1 locomotion.

Loop structure per iteration
─────────────────────────────
  1. PROBE     — measure survival_rate on slope_up + stairs_up with current model
  2. EVALUATE  — is combined ≥ TARGET?  If yes → SUCCESS, exit.
  3. DEVIL'S ADVOCATE — argue against the proposed fix; decide if it makes sense
  4. FINETUNE  — run finetune_terrain.py for STEP_SIZE more steps
  5. RERUN     — go to step 1

Stop conditions
───────────────
  • Combined survival (slope_up + stairs_up, d=0.3) ≥ TARGET (default 0.60)
  • Max iterations reached (default 5)
  • Flat terrain survival drops below FLAT_MIN (0.60) — catastrophic forgetting

Usage
─────
    python3 scripts/agentic_terrain_loop.py [--max-iter 5] [--target 0.6]
"""
import os
import sys
import json
import time
import subprocess
import argparse
import warnings
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


# ─── constants ───────────────────────────────────────────────────────────────
TARGET_COMBINED   = 0.60   # slope + stairs combined survival needed to stop
FLAT_MIN          = 0.50   # flat survival below this → catastrophic forgetting
STEP_SIZE_DEFAULT = 250_000  # timesteps per fine-tune iteration
N_ENVS_DEFAULT    = 4
PROBE_EPISODES    = 10
LOG_FILE          = "logs/agentic_terrain_loop.jsonl"


# ─── helpers ─────────────────────────────────────────────────────────────────

def _load_policy(path: str):
    from stable_baselines3 import PPO
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PPO.load(path, device="cpu")


def probe(model_path: str) -> dict:
    """Run survival probe on slope_up, stairs_up and flat.  Returns dict."""
    # Import here so we can use the PROJECT_ROOT path
    sys.path.insert(0, PROJECT_ROOT)
    from scripts.finetune_terrain import probe_terrain
    model = _load_policy(model_path)

    results = {}
    for terrain, diff in [("slope_up", 0.3), ("stairs_up", 0.3), ("flat", 0.0)]:
        r = probe_terrain(model, n_episodes=PROBE_EPISODES,
                          terrain_type=terrain, difficulty=diff)
        results[terrain] = r
        print(f"  {terrain:12s}  survival={r['survival_rate']:.2f}  vx={r['mean_vx']:.3f}")
    combined = (results["slope_up"]["survival_rate"] +
                results["stairs_up"]["survival_rate"]) / 2.0
    results["combined"] = combined
    return results


def devils_advocate(results: dict, iteration: int, step_size: int) -> tuple[bool, str]:
    """Argue against the fine-tuning plan; return (proceed: bool, reason: str).

    Returns proceed=False if we detect:
      • Flat terrain regressing severely (< FLAT_MIN)
      • Terrain scores already meet target (no need to train more)
      • We've exceeded max reasonable compute budget
    """
    flat_ok   = results.get("flat",  {}).get("survival_rate", 1.0) >= FLAT_MIN
    combined  = results.get("combined", 0.0)
    target_ok = combined >= TARGET_COMBINED

    if target_ok:
        return False, f"TARGET MET (combined={combined:.2f} ≥ {TARGET_COMBINED}). Stop."

    if not flat_ok:
        flat_sr = results.get("flat", {}).get("survival_rate", 0.0)
        return False, (
            f"CATASTROPHIC FORGETTING: flat survival={flat_sr:.2f} < {FLAT_MIN}. "
            "Further training without flat env mix will hurt base skill."
        )

    if iteration >= 4 and combined < 0.20:
        return False, (
            f"DIMINISHING RETURNS: After {iteration} iterations combined={combined:.2f}. "
            "Curriculum or reward shaping changes needed — fine-tuning alone insufficient."
        )

    reason = (
        f"Proceeding: combined={combined:.2f} < {TARGET_COMBINED}, "
        f"flat={results.get('flat', {}).get('survival_rate', 0.0):.2f} OK, "
        f"iteration={iteration}."
    )
    return True, reason


def log_event(record: dict):
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_finetune(step_size: int, n_envs: int, finetune_lr: float,
                 resume: str, checkpoint_dir: str) -> int:
    """Run finetune_terrain.py as a subprocess, return exit code."""
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "finetune_terrain.py"),
        "--resume",         resume,
        "--steps",          str(step_size),
        "--n-envs",         str(n_envs),
        "--finetune-lr",    str(finetune_lr),
        "--checkpoint-dir", checkpoint_dir,
        "--device",         "cpu",
    ]
    print(f"\n[loop] Running: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


# ─── main loop ────────────────────────────────────────────────────────────────

def run_loop(args):
    model_path   = args.resume
    step_size    = args.step_size
    n_envs       = args.n_envs
    finetune_lr  = args.finetune_lr
    max_iter     = args.max_iter
    ckpt_dir     = args.checkpoint_dir

    print("=" * 60)
    print("AGENTIC TERRAIN REPAIR LOOP")
    print(f"  model       : {model_path}")
    print(f"  target      : combined survival ≥ {TARGET_COMBINED}")
    print(f"  step_size   : {step_size:,} per iteration")
    print(f"  max_iter    : {max_iter}")
    print("=" * 60)

    for iteration in range(max_iter):
        ts = datetime.now().isoformat()
        print(f"\n{'─'*60}")
        print(f"ITERATION {iteration+1}/{max_iter}   [{ts}]")
        print(f"{'─'*60}")

        # ── STEP 1: PROBE ────────────────────────────────────────────────
        print("\n[1] PROBE — measuring terrain survival...")
        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found: {model_path}")
            break
        results = probe(model_path)
        combined = results["combined"]
        log_event({"iter": iteration, "phase": "probe", "ts": ts,
                   "results": results, "model": model_path})

        # ── STEP 2: EVALUATE ─────────────────────────────────────────────
        print(f"\n[2] EVALUATE — combined={combined:.2f} (target={TARGET_COMBINED})")
        if combined >= TARGET_COMBINED:
            print(f"\n✓✓✓  TARGET REACHED  (combined={combined:.2f} ≥ {TARGET_COMBINED})")
            log_event({"iter": iteration, "phase": "success", "combined": combined})
            break

        # ── STEP 3: DEVIL'S ADVOCATE ─────────────────────────────────────
        print("\n[3] DEVIL'S ADVOCATE — evaluating whether to continue...")
        proceed, reason = devils_advocate(results, iteration, step_size)
        print(f"    → {reason}")
        log_event({"iter": iteration, "phase": "devils_advocate",
                   "proceed": proceed, "reason": reason})

        if not proceed:
            print("\n[loop] Devil's advocate vetoed further training. Stopping.")
            break

        # ── STEP 4: FINE-TUNE ────────────────────────────────────────────
        print(f"\n[4] FINE-TUNE — {step_size:,} more steps at LR={finetune_lr}...")
        iter_ckpt_dir = os.path.join(ckpt_dir, f"iter_{iteration+1}")
        rc = run_finetune(step_size, n_envs, finetune_lr, model_path, iter_ckpt_dir)
        log_event({"iter": iteration, "phase": "finetune",
                   "rc": rc, "ckpt_dir": iter_ckpt_dir})

        if rc != 0:
            print(f"[loop] Fine-tune exited with code {rc}. Stopping.")
            break

        # The finetune script promotes best to checkpoints/best/best_model.zip
        # if combined ≥ 0.5; otherwise use its best_terrain_ft.zip directly.
        promoted = "checkpoints/best/best_model.zip"
        ft_best  = os.path.join(iter_ckpt_dir, "best_terrain_ft.zip")
        ft_final = os.path.join(iter_ckpt_dir, "terrain_ft_final.zip")

        # Select best checkpoint for next iteration
        for candidate in [promoted, ft_best, ft_final]:
            if os.path.exists(candidate):
                model_path = candidate
                print(f"[loop] Next iteration will use: {model_path}")
                break
        else:
            print("[loop] No fine-tuned checkpoint found. Stopping.")
            break

        # ── STEP 5: RERUN (implicit — next loop iteration) ──────────────
        print("\n[5] RERUN → looping back to probe...")

    print("\n" + "=" * 60)
    print("AGENTIC LOOP COMPLETE")
    print(f"Final model: {model_path}")
    print(f"Log: {LOG_FILE}")
    print("=" * 60)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    global TARGET_COMBINED
    parser = argparse.ArgumentParser(
        description="Agentic terrain repair loop for Go1 locomotion policy"
    )
    parser.add_argument("--resume",        default="checkpoints/best/best_model.zip")
    parser.add_argument("--step-size",     type=int,   default=STEP_SIZE_DEFAULT)
    parser.add_argument("--n-envs",        type=int,   default=N_ENVS_DEFAULT)
    parser.add_argument("--finetune-lr",   type=float, default=5e-5)
    parser.add_argument("--max-iter",      type=int,   default=5)
    parser.add_argument("--target",        type=float, default=TARGET_COMBINED)
    parser.add_argument("--checkpoint-dir", default="checkpoints/terrain_ft")
    args = parser.parse_args()
    TARGET_COMBINED = args.target
    run_loop(args)