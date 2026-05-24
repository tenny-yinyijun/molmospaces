#!/usr/bin/env bash
# In-process MolmoBot eval on a MolmoSpaces benchmark.
#
# Sim eval is in-process (no websocket server) — runs molmospaces' eval_main
# with olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig using
# MolmoBot's venv (which ships a pinned molmo_spaces + olmo).
#
# Usage:
#   bash bash_scripts/run_molmobot_local.sh                                # default bench
#   BENCH_DIR=/abs/path/to/<bench>_json_benchmark bash bash_scripts/run_molmobot_local.sh
#   MAX_EPISODES=10 bash bash_scripts/run_molmobot_local.sh -- --idx 734  # extra eval_main flags
#
# Env knobs:
#   MOLMOBOT_DIR     MolmoBot/MolmoBot root (default: ../MolmoBot/MolmoBot)
#   MOLMOBOT_CKPT    weights dir (default: ~/.cache/molmobot/MolmoBot-DROID)
#   BENCH_DIR        benchmark directory
#   TASK_HORIZON     --task_horizon_steps (default 600, per MolmoBot README)
#   MAX_EPISODES     cap episodes (default: all)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOLMOBOT_DIR="${MOLMOBOT_DIR:-$REPO/../MolmoBot/MolmoBot}"
MOLMOBOT_VENV="$MOLMOBOT_DIR/.venv"
MOLMOBOT_CKPT="${MOLMOBOT_CKPT:-$HOME/.cache/molmobot/MolmoBot-DROID}"
# Default benchmark: FrankaPickDroidMiniBench (matches sbatch_corruption_sweep.sh).
# Cache hash dir is derived from the molmospaces repo path; glob to stay portable.
ASSETS_ROOT="${MLSPACES_ASSETS_DIR:-$HOME/.cache/molmospaces}"
BENCH_DIR="${BENCH_DIR:-$(ls -d "$ASSETS_ROOT"/assets/*/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_* 2>/dev/null | tail -1)}"
TASK_HORIZON="${TASK_HORIZON:-600}"

[[ -d "$MOLMOBOT_VENV"   ]] || { echo "[run_molmobot_local] venv missing: $MOLMOBOT_VENV (run 'cd $MOLMOBOT_DIR && uv sync --extra eval')" >&2; exit 1; }
[[ -f "$MOLMOBOT_CKPT/model.pt" ]] || { echo "[run_molmobot_local] weights missing: $MOLMOBOT_CKPT/model.pt" >&2; exit 1; }
[[ -d "$BENCH_DIR"       ]] || { echo "[run_molmobot_local] benchmark missing: $BENCH_DIR" >&2; exit 1; }

# Pin JAX to CPU (molmobot uses torch on GPU); keep MuJoCo on EGL; stay offline.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_CONSOLE=off
export PYTHONUNBUFFERED=1

EXTRA_ARGS=()
[[ -n "${MAX_EPISODES:-}" ]] && EXTRA_ARGS+=(--max_episodes "$MAX_EPISODES")

echo "[run_molmobot_local] checkpoint : $MOLMOBOT_CKPT"
echo "[run_molmobot_local] benchmark  : $BENCH_DIR"
echo "[run_molmobot_local] horizon    : $TASK_HORIZON"

# Run from MolmoBot dir so olmo and its pinned molmo_spaces resolve from this venv.
cd "$MOLMOBOT_DIR"

"$MOLMOBOT_VENV/bin/python" -X faulthandler \
    -m molmo_spaces.evaluation.eval_main \
    olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig \
    --checkpoint_path "$MOLMOBOT_CKPT" \
    --benchmark_dir "$BENCH_DIR" \
    --task_horizon_steps "$TASK_HORIZON" \
    --output_dir "$REPO/eval_output" \
    --no_wandb \
    "${EXTRA_ARGS[@]}" "$@"

# Collect per-episode videos labeled _success/_failure.
LATEST=$(ls -td "$REPO/eval_output/FrankaState8ClampAbsPosConfig"/*/ 2>/dev/null | head -1)
[[ -n "$LATEST" ]] || { echo "[run_molmobot_local] no eval output dir under eval_output/FrankaState8ClampAbsPosConfig/" >&2; exit 1; }
OUT_DIR="$REPO/bash_scripts/eval_videos/molmobot"
"$MOLMOBOT_VENV/bin/python" "$REPO/scripts/collect_eval_videos.py" \
    --eval_dir "$LATEST" \
    --out_dir "$OUT_DIR"
echo "[run_molmobot_local] eval output : $LATEST"
echo "[run_molmobot_local] videos      : $OUT_DIR"
