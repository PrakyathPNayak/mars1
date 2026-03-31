"""
Rich-based live training dashboard with TensorBoard integration.

Provides real-time visualization of:
  - Episode rewards (rolling mean + sparkline)
  - Policy/value loss curves
  - Per-component reward breakdown
  - Learning rate, entropy, clip fraction
  - Episode length statistics

Usage:
  python3 -m src.visualization.live_dashboard                        # auto-detect log dir
  python3 -m src.visualization.live_dashboard --log-dir logs/training
  python3 -m src.visualization.live_dashboard --tensorboard           # also launch TensorBoard
"""
import os
import sys
import csv
import io
import time
import argparse
import subprocess
import signal
from pathlib import Path
from collections import deque

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box


# ── Sparkline helper ────────────────────────────────────────────

SPARK_CHARS = "▁▂▃▄▅▆▇█"

def sparkline(values, width=40):
    """Render a list of floats as a unicode sparkline string."""
    if not values:
        return ""
    vals = list(values)[-width:]
    lo, hi = min(vals), max(vals)
    rng = hi - lo if hi != lo else 1.0
    return "".join(SPARK_CHARS[min(int((v - lo) / rng * 7), 7)] for v in vals)


def rolling_mean(values, window=100):
    """Compute rolling mean of last `window` values."""
    if not values:
        return 0.0
    recent = list(values)[-window:]
    return sum(recent) / len(recent)


# ── TensorBoard event reader ───────────────────────────────────

def read_tb_scalars(event_dir):
    """Read scalar summaries from TensorBoard event files.

    Returns dict of {tag: [(wall_time, step, value), ...]}
    """
    scalars = {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        for subdir in sorted(Path(event_dir).rglob("events.out.tfevents.*")):
            ea = EventAccumulator(str(subdir.parent), size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(tag)
                if tag not in scalars:
                    scalars[tag] = []
                for e in events:
                    scalars[tag].append((e.wall_time, e.step, e.value))
    except Exception:
        pass
    return scalars


# ── Monitor CSV reader ──────────────────────────────────────────

def read_monitor_csv(log_dir):
    """Read SB3 monitor.csv files. Returns (rewards, lengths, timestamps)."""
    rewards, lengths, timestamps = [], [], []
    for f in sorted(Path(log_dir).glob("*.monitor.csv")):
        try:
            text = f.read_text().strip()
            lines = text.split("\n")
            if len(lines) < 3:
                continue
            reader = csv.DictReader(io.StringIO("\n".join(lines[1:])))
            for row in reader:
                rewards.append(float(row["r"]))
                lengths.append(float(row["l"]))
                timestamps.append(float(row["t"]))
        except Exception:
            continue
    return rewards, lengths, timestamps


# ── Reward component log reader ─────────────────────────────────

def read_reward_components(log_dir):
    """Read per-component reward CSV if it exists."""
    path = Path(log_dir) / "reward_components.csv"
    components = {}
    if not path.exists():
        return components
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in row:
                    if key == "step":
                        continue
                    if key not in components:
                        components[key] = []
                    components[key].append(float(row[key]))
    except Exception:
        pass
    return components


# ── Dashboard builder ───────────────────────────────────────────

def build_dashboard(log_dir, tb_scalars, rewards, lengths, timestamps, components):
    """Build a rich Layout with all training metrics."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )

    # ── Header
    header_text = Text("  MIT Mini Cheetah — Live Training Dashboard", style="bold white on blue")
    layout["header"].update(Panel(header_text, box=box.HEAVY))

    # ── Left panel: reward curves + loss
    left_parts = []

    # Reward sparkline
    if rewards:
        rm = rolling_mean(rewards, 100)
        spark = sparkline(rewards)
        n = len(rewards)
        mx = max(rewards)
        mn = min(rewards)
        left_parts.append(
            f"[bold cyan]Episode Rewards[/] ({n} episodes)\n"
            f"  Mean(100): [green]{rm:+.1f}[/]  Max: {mx:+.1f}  Min: {mn:+.1f}\n"
            f"  [dim]{spark}[/]"
        )
    else:
        left_parts.append("[yellow]No episode data yet. Waiting for training...[/]")

    # Episode length sparkline
    if lengths:
        avg_len = rolling_mean(lengths, 100)
        spark_l = sparkline(lengths)
        left_parts.append(
            f"[bold cyan]Episode Length[/]\n"
            f"  Mean(100): {avg_len:.0f}  Max: {max(lengths):.0f}\n"
            f"  [dim]{spark_l}[/]"
        )

    # TensorBoard scalars: policy loss, value loss, entropy, etc.
    loss_tags = {
        "train/policy_gradient_loss": "Policy Loss",
        "train/value_loss": "Value Loss",
        "train/entropy_loss": "Entropy",
        "train/clip_fraction": "Clip Frac",
        "train/approx_kl": "Approx KL",
        "train/learning_rate": "LR",
    }
    loss_lines = []
    for tag, label in loss_tags.items():
        if tag in tb_scalars and tb_scalars[tag]:
            vals = [v for _, _, v in tb_scalars[tag]]
            latest = vals[-1]
            spark_v = sparkline(vals[-40:], width=30)
            loss_lines.append(f"  {label:16s}: {latest:+.6f}  [dim]{spark_v}[/]")
    if loss_lines:
        left_parts.append("[bold cyan]Training Losses[/]\n" + "\n".join(loss_lines))

    layout["left"].update(Panel("\n\n".join(left_parts), title="Training Progress", border_style="cyan"))

    # ── Right panel: reward components + summary
    right_parts = []

    # Reward component breakdown
    if components:
        comp_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        comp_table.add_column("Component", style="white")
        comp_table.add_column("Mean(100)", justify="right")
        comp_table.add_column("Trend", justify="left")
        for name, vals in components.items():
            m = rolling_mean(vals, 100)
            sp = sparkline(vals[-30:], width=20)
            color = "green" if m > 0 else "red"
            comp_table.add_row(name, f"[{color}]{m:+.3f}[/]", f"[dim]{sp}[/]")
        right_parts.append(comp_table)
    else:
        right_parts.append("[dim]Reward component logging not active.\nAdd RewardComponentLogger callback to training.[/]")

    # Summary stats
    summary_table = Table(box=box.SIMPLE, show_header=False)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")
    if rewards:
        summary_table.add_row("Episodes", str(len(rewards)))
        summary_table.add_row("Best Reward", f"{max(rewards):+.1f}")
        summary_table.add_row("Mean(100)", f"{rolling_mean(rewards, 100):+.1f}")
        summary_table.add_row("Std(100)", f"{std_recent(rewards, 100):.1f}")
    if timestamps:
        elapsed = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        eps_per_min = len(rewards) / (elapsed / 60) if elapsed > 60 else 0
        summary_table.add_row("Elapsed", format_duration(elapsed))
        summary_table.add_row("Eps/min", f"{eps_per_min:.1f}")

    # TensorBoard extras
    for tag in ["rollout/ep_rew_mean", "rollout/ep_len_mean"]:
        if tag in tb_scalars and tb_scalars[tag]:
            vals = [v for _, _, v in tb_scalars[tag]]
            label = "TB rew mean" if "rew" in tag else "TB len mean"
            summary_table.add_row(label, f"{vals[-1]:.1f}")

    right_parts.append(summary_table)

    right_content = "\n".join(str(p) if not isinstance(p, str) else p for p in right_parts)
    # Use renderables list for mixed types
    from rich.console import Group
    layout["right"].update(Panel(Group(*right_parts), title="Components & Stats", border_style="magenta"))

    # ── Footer
    layout["footer"].update(
        Panel(
            "[dim]Press Ctrl+C to quit | Auto-refreshes every 5s | "
            "Use --tensorboard flag to also launch TensorBoard UI[/]",
            box=box.ROUNDED,
        )
    )

    return layout


def std_recent(values, window=100):
    if not values:
        return 0.0
    recent = values[-window:]
    m = sum(recent) / len(recent)
    return (sum((v - m) ** 2 for v in recent) / len(recent)) ** 0.5


def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


# ── Main loop ───────────────────────────────────────────────────

def run_dashboard(log_dir="logs/training", refresh_interval=5, launch_tb=False):
    """Run the live dashboard in the terminal."""
    console = Console()
    log_dir = str(log_dir)

    if not Path(log_dir).exists():
        console.print(f"[red]Log directory not found: {log_dir}[/]")
        console.print("[yellow]Start training first, or specify --log-dir[/]")
        return

    tb_proc = None
    if launch_tb:
        try:
            tb_proc = subprocess.Popen(
                ["tensorboard", "--logdir", log_dir, "--port", "6006", "--bind_all"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            console.print("[green]TensorBoard launched at http://localhost:6006[/]")
        except FileNotFoundError:
            console.print("[yellow]tensorboard command not found. Install with: pip install tensorboard[/]")

    console.print(f"[bold]Monitoring: {log_dir}[/]")
    console.print(f"[dim]Refresh every {refresh_interval}s. Ctrl+C to stop.[/]\n")

    try:
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                tb_scalars = read_tb_scalars(log_dir)
                rewards, lengths, timestamps = read_monitor_csv(log_dir)
                components = read_reward_components(log_dir)
                dashboard = build_dashboard(log_dir, tb_scalars, rewards, lengths, timestamps, components)
                live.update(dashboard)
                time.sleep(refresh_interval)
    except KeyboardInterrupt:
        console.print("\n[bold]Dashboard stopped.[/]")
    finally:
        if tb_proc:
            tb_proc.terminate()
            tb_proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Live training dashboard")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Training log directory (auto-detects)")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Also launch TensorBoard on port 6006")
    args = parser.parse_args()

    # Auto-detect log directory
    if args.log_dir:
        log_dir = args.log_dir
    else:
        for candidate in ["logs/training_advanced", "logs/training"]:
            if Path(candidate).exists():
                log_dir = candidate
                break
        else:
            log_dir = "logs/training"

    run_dashboard(log_dir=log_dir, refresh_interval=args.refresh, launch_tb=args.tensorboard)


if __name__ == "__main__":
    main()
