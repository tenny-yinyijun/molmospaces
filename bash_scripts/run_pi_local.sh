#!/usr/bin/env bash
# In-process pi0 / pi05 / pi0_fast eval on a MolmoSpaces benchmark — NO websocket server.
#
# How: PI_Policy._prepare_local_model loads the openpi checkpoint directly when
# PiPolicyConfig.remote_config is None. PiLocalPolicyEvalConfig sets that. We
# run molmospaces' eval_main with openpi's uv-managed venv (which has openpi
# importable plus molmospaces editable-installed).
#
# Usage:
#   bash bash_scripts/run_pi_local.sh                                 # pi05, default bench
#   POLICY=pi0 bash bash_scripts/run_pi_local.sh                      # pi0
#   BENCH_DIR=/abs/path/to/<bench>_json_benchmark bash bash_scripts/run_pi_local.sh
#   MAX_EPISODES=10 bash bash_scripts/run_pi_local.sh -- --idx 734    # extra eval_main flags
#
# First time only (adds the few molmospaces deps missing from openpi's venv):
#   bash bash_scripts/run_pi_local.sh --install-deps
#
# Env knobs:
#   POLICY            pi05 (default) | pi0 | pi0_fast
#   CKPT_DIR          override checkpoint directory
#   BENCH_DIR         override benchmark directory
#   TASK_HORIZON      --task_horizon_steps (default 500)
#   MAX_EPISODES      cap episodes (default: all)
#   OPENPI_DIR        openpi repo root (default: ../openpi)
#   OPENPI_DATA_HOME  required — contains checkpoints/<CKPT_NAME>/

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$REPO/../openpi}"
OPENPI_VENV="$OPENPI_DIR/.venv"

[[ -d "$OPENPI_VENV" ]] || { echo "[run_pi_local] openpi venv missing: $OPENPI_VENV (run 'cd $OPENPI_DIR && uv sync')" >&2; exit 1; }

# Deps that molmo_spaces needs but openpi's lockfile doesn't pull in.
EXTRA_DEPS=(scikit-image trimesh openai anthropic)

if [[ "${1:-}" == "--install-deps" ]]; then
    echo "[run_pi_local] installing into $OPENPI_VENV: ${EXTRA_DEPS[*]}"
    "$OPENPI_VENV/bin/pip" install --quiet "${EXTRA_DEPS[@]}"
    exit 0
fi

if ! "$OPENPI_VENV/bin/python" -c 'import skimage, trimesh, openai, anthropic' 2>/dev/null; then
    echo "[run_pi_local] missing deps in $OPENPI_VENV. Run once:" >&2
    echo "    bash $0 --install-deps" >&2
    exit 1
fi

POLICY="${POLICY:-pi05}"
case "$POLICY" in
    pi0)      CKPT_NAME=pi0_droid_jointpos ;;
    pi05)     CKPT_NAME=pi05_droid_jointpos ;;
    pi0_fast) CKPT_NAME=pi0_fast_droid_jointpos ;;
    *) echo "[run_pi_local] unknown POLICY=$POLICY (use pi0 | pi05 | pi0_fast)" >&2; exit 1 ;;
esac

: "${OPENPI_DATA_HOME:?OPENPI_DATA_HOME is not set; export it (e.g. /scratch/gpfs/AM43/yy4041/.cache/openpi)}"
CKPT_DIR="${CKPT_DIR:-$OPENPI_DATA_HOME/checkpoints/$CKPT_NAME}"
# Default benchmark: FrankaPickDroidMiniBench (matches sbatch_corruption_sweep.sh).
# Cache hash dir is derived from the molmospaces repo path; glob to stay portable.
ASSETS_ROOT="${MLSPACES_ASSETS_DIR:-$HOME/.cache/molmospaces}"
BENCH_DIR="${BENCH_DIR:-$(ls -d "$ASSETS_ROOT"/assets/*/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_* 2>/dev/null | tail -1)}"
TASK_HORIZON="${TASK_HORIZON:-500}"

[[ -d "$CKPT_DIR"   ]] || { echo "[run_pi_local] checkpoint missing: $CKPT_DIR" >&2; exit 1; }
[[ -d "$BENCH_DIR"  ]] || { echo "[run_pi_local] benchmark missing: $BENCH_DIR" >&2; exit 1; }

cd "$REPO"

# Keep JAX off the EGL framebuffer pages (mirrors run_pi_inference.sh) so render
# and inference can coexist on the same GPU without aborting mjr_readPixels.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_CONSOLE=off
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60

EXTRA_ARGS=()
[[ -n "${MAX_EPISODES:-}" ]] && EXTRA_ARGS+=(--max_episodes "$MAX_EPISODES")

echo "[run_pi_local] policy     : $POLICY ($CKPT_NAME)"
echo "[run_pi_local] checkpoint : $CKPT_DIR"
echo "[run_pi_local] benchmark  : $BENCH_DIR"
echo "[run_pi_local] horizon    : $TASK_HORIZON"

"$OPENPI_VENV/bin/python" -X faulthandler \
    -m molmo_spaces.evaluation.eval_main \
    molmo_spaces.evaluation.configs.evaluation_configs:PiLocalPolicyEvalConfig \
    --checkpoint_path "$CKPT_DIR" \
    --benchmark_dir "$BENCH_DIR" \
    --task_horizon_steps "$TASK_HORIZON" \
    --no_wandb \
    "${EXTRA_ARGS[@]}" "$@"

# Collect per-episode videos labeled _success/_failure.
LATEST=$(ls -td "$REPO/eval_output/PiLocalPolicyEvalConfig"/*/ 2>/dev/null | head -1)
[[ -n "$LATEST" ]] || { echo "[run_pi_local] no eval output dir under eval_output/PiLocalPolicyEvalConfig/" >&2; exit 1; }
OUT_DIR="$REPO/bash_scripts/eval_videos/$POLICY"
"$OPENPI_VENV/bin/python" "$REPO/scripts/collect_eval_videos.py" \
    --eval_dir "$LATEST" \
    --out_dir "$OUT_DIR"
echo "[run_pi_local] eval output : $LATEST"
echo "[run_pi_local] videos      : $OUT_DIR"
