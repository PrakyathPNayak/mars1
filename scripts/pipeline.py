"""Unified training pipeline: Standard MLP PPO → Hierarchical Transformer+MoE.

Stage 1: Train a standard MLP PPO policy from scratch.
Stage 2: Use the Stage 1 best model as an expert for hierarchical BC→Transformer+MoE training.

All outputs are consolidated into a single timestamped run folder:

    runs/{run_id}/
        mlp_final.zip            ← Stage 1 final model
        mlp_best.zip             ← Stage 1 best evaluated model
        mlp_vec_normalize.pkl    ← VecNormalize stats (needed for inference)
        hierarchical_final.zip   ← Stage 2 final model
        hierarchical_best.zip    ← Stage 2 best evaluated model
        training_summary.json    ← combined training metadata

Usage:
    python3 scripts/pipeline.py
    python3 scripts/pipeline.py --mlp-steps 5000000 --hier-steps 10000000
    python3 scripts/pipeline.py --run-id my_experiment --device cuda
    python3 scripts/pipeline.py --skip-mlp --expert checkpoints/best/best_model.zip
"""
import argparse
import json
import os
import shutil
import sys
import time
import types
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ─── helpers ──────────────────────────────────────────────────────────────────

def _copy(src: Path, dst: Path, label: str = "") -> bool:
    """Copy src → dst, trying .zip extension if bare path not found."""
    if not src.exists():
        src_zip = src.with_suffix(".zip") if src.suffix != ".zip" else src
        if src_zip.exists():
            src = src_zip
        else:
            print(f"  [WARN] Not found, skipping: {src}")
            return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    tag = f"  [{label}]" if label else " "
    print(f"{tag} {src.name} → {dst.relative_to(_PROJECT_ROOT)}")
    return True


def _build_mlp_args(args, run_dir: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        total_steps=args.mlp_steps,
        n_envs=args.n_envs,
        device=args.device,
        resume=None,
        ckpt_dir=str(run_dir / "mlp_training"),
        log_dir=str(run_dir / "logs_mlp"),
    )


def _build_hier_args(
    args, run_dir: Path, expert_path: str, vec_normalize_path: str | None
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        expert=expert_path,
        n_expert_episodes=args.n_expert_episodes,
        bc_epochs=args.bc_epochs,
        bc_lr=args.bc_lr,
        bc_batch=args.bc_batch,
        total_steps=args.hier_steps,
        n_envs=args.n_envs,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_experts=args.n_experts,
        history_len=args.history_len,
        device=args.device,
        verbose=args.verbose,
        ckpt_dir=str(run_dir / "hierarchical_training"),
        log_dir=str(run_dir / "logs_hierarchical"),
        vec_normalize=vec_normalize_path,
    )


# ─── main pipeline ──────────────────────────────────────────────────────────

def run_pipeline(args):
    os.chdir(_PROJECT_ROOT)

    from src.training import train as train_mod
    from src.training import train_hierarchical as hier_mod

    # Create the consolidated run output folder.
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _PROJECT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("PIPELINE: MLP PPO  →  Hierarchical Transformer+MoE")
    print(f"Run ID : {run_id}")
    print(f"Output : {run_dir}")
    print(f"{'='*60}\n")

    summary: dict = {
        "run_id": run_id,
        "started": datetime.now().isoformat(),
        "stages": {},
    }

    # ──────────────────────────────────────────────────────────────────────────
    #  Stage 1: Standard MLP PPO Training
    # ──────────────────────────────────────────────────────────────────────────
    if args.skip_mlp:
        # User passes a pre-trained expert; skip Stage 1.
        expert_path = str(args.expert)
        vec_normalize_path = args.vec_normalize
        print("[Stage 1] Skipped (--skip-mlp).  Expert:", expert_path)
        if vec_normalize_path is None:
            # Try the standard location next to the expert.
            candidate = Path(expert_path).parent / "vec_normalize.pkl"
            if candidate.exists():
                vec_normalize_path = str(candidate)
                print(f"           Auto-detected VecNormalize: {vec_normalize_path}")
        summary["stages"]["stage1_mlp"] = {
            "skipped": True,
            "expert": expert_path,
            "vec_normalize": vec_normalize_path,
        }
    else:
        print("=" * 60)
        print("STAGE 1: Standard MLP PPO Training")
        print("=" * 60)
        t0 = time.time()

        mlp_args = _build_mlp_args(args, run_dir)
        # Pre-create expected dirs so EvalCallback doesn't fail.
        (Path(mlp_args.ckpt_dir) / "best").mkdir(parents=True, exist_ok=True)
        Path(mlp_args.log_dir).mkdir(parents=True, exist_ok=True)

        mlp_final_internal = train_mod.train(mlp_args)

        elapsed1 = time.time() - t0
        mlp_ckpt_dir = Path(mlp_args.ckpt_dir)

        # Resolve source paths (SB3 saves without .zip, file has .zip).
        mlp_final_src = Path(mlp_final_internal)
        if not mlp_final_src.exists():
            mlp_final_src = mlp_final_src.with_suffix(".zip")
        mlp_best_src = mlp_ckpt_dir / "best" / "best_model.zip"
        mlp_norm_src = mlp_ckpt_dir / "vec_normalize.pkl"

        print("\n[Stage 1] Consolidating MLP outputs ...")
        _copy(mlp_final_src, run_dir / "mlp_final.zip", "mlp_final")
        _copy(mlp_best_src, run_dir / "mlp_best.zip", "mlp_best")
        if mlp_norm_src.exists():
            shutil.copy2(mlp_norm_src, run_dir / "mlp_vec_normalize.pkl")
            print(f"  [mlp_norm] vec_normalize.pkl → mlp_vec_normalize.pkl")

        summary["stages"]["stage1_mlp"] = {
            "elapsed_seconds": elapsed1,
            "total_steps": args.mlp_steps,
            "final_model": str(run_dir / "mlp_final.zip"),
            "best_model": str(run_dir / "mlp_best.zip"),
            "vec_normalize": str(run_dir / "mlp_vec_normalize.pkl"),
        }

        # Prefer best model as expert; fall back to final if not available.
        if (run_dir / "mlp_best.zip").exists():
            expert_path = str(run_dir / "mlp_best.zip")
        else:
            expert_path = str(run_dir / "mlp_final.zip")
            print("[Stage 1] No best model found — using final model as expert")

        vec_normalize_path = (
            str(run_dir / "mlp_vec_normalize.pkl")
            if (run_dir / "mlp_vec_normalize.pkl").exists()
            else None
        )

        print(f"\n[Stage 1] Done in {elapsed1:.0f}s.  Expert: {expert_path}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Stage 2: Hierarchical BC → Transformer+MoE PPO Training
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Hierarchical BC → Transformer+MoE PPO Training")
    print("=" * 60)
    t0 = time.time()

    hier_args = _build_hier_args(args, run_dir, expert_path, vec_normalize_path)
    # Pre-create dirs.
    (Path(hier_args.ckpt_dir) / "best").mkdir(parents=True, exist_ok=True)
    Path(hier_args.log_dir).mkdir(parents=True, exist_ok=True)

    hier_final_internal = hier_mod.train(hier_args)

    elapsed2 = time.time() - t0
    hier_ckpt_dir = Path(hier_args.ckpt_dir)

    hier_final_src = Path(hier_final_internal)
    if not hier_final_src.exists():
        hier_final_src = hier_final_src.with_suffix(".zip")
    hier_best_src = hier_ckpt_dir / "best" / "best_model.zip"

    print("\n[Stage 2] Consolidating hierarchical outputs ...")
    _copy(hier_final_src, run_dir / "hierarchical_final.zip", "hier_final")
    _copy(hier_best_src, run_dir / "hierarchical_best.zip", "hier_best")

    summary["stages"]["stage2_hierarchical"] = {
        "elapsed_seconds": elapsed2,
        "total_steps": args.hier_steps,
        "expert": expert_path,
        "final_model": str(run_dir / "hierarchical_final.zip"),
        "best_model": str(run_dir / "hierarchical_best.zip"),
    }

    # ──────────────────────────────────────────────────────────────────────────
    #  Write consolidated training summary
    # ──────────────────────────────────────────────────────────────────────────
    total_elapsed = sum(
        s.get("elapsed_seconds", 0) for s in summary["stages"].values()
    )
    summary["finished"] = datetime.now().isoformat()
    summary["total_elapsed_seconds"] = total_elapsed
    summary["output_folder"] = str(run_dir)
    summary["models"] = {
        "mlp_final": str(run_dir / "mlp_final.zip"),
        "mlp_best": str(run_dir / "mlp_best.zip"),
        "mlp_vec_normalize": str(run_dir / "mlp_vec_normalize.pkl"),
        "hierarchical_final": str(run_dir / "hierarchical_final.zip"),
        "hierarchical_best": str(run_dir / "hierarchical_best.zip"),
    }

    summary_path = run_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ──────────────────────────────────────────────────────────────────────────
    #  Final report
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    h, m = divmod(int(total_elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"Total time : {h}h {m}m {s}s")
    print(f"Output dir : {run_dir}")
    print("Models:")
    for name, path in summary["models"].items():
        mark = "✓" if Path(path).exists() else "✗"
        print(f"  {mark}  {name:<30s}  {Path(path).name}")
    print(f"\nSummary    : {summary_path}")

    return str(run_dir)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full training pipeline: MLP PPO → Hierarchical Transformer+MoE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Run identity ──
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run identifier (default: YYYYMMDD_HHMMSS timestamp)",
    )

    # ── Stage 1: MLP PPO ──
    parser.add_argument(
        "--mlp-steps", type=int, default=5_000_000,
        help="Total env steps for MLP PPO stage",
    )
    parser.add_argument(
        "--skip-mlp", action="store_true",
        help="Skip Stage 1 and use --expert directly as the base model",
    )
    parser.add_argument(
        "--expert", type=str, default=None,
        help="Path to pre-trained expert .zip (required when --skip-mlp)",
    )
    parser.add_argument(
        "--vec-normalize", type=str, default=None,
        help="Path to VecNormalize .pkl when --skip-mlp is set",
    )

    # ── Stage 2: Hierarchical PPO ──
    parser.add_argument(
        "--hier-steps", type=int, default=10_000_000,
        help="Total env steps for hierarchical PPO stage",
    )
    parser.add_argument("--n-expert-episodes", type=int, default=200)
    parser.add_argument("--bc-epochs", type=int, default=50)
    parser.add_argument("--bc-lr", type=float, default=5e-4)
    parser.add_argument("--bc-batch", type=int, default=256)

    # ── Architecture (hierarchical) ──
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--history-len", type=int, default=16)

    # ── Shared ──
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device: cpu | cuda | auto",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.skip_mlp and not args.expert:
        parser.error("--skip-mlp requires --expert <path>")

    run_pipeline(args)


if __name__ == "__main__":
    main()
