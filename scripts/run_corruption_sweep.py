"""Sweep N molmobot runs of one episode with increasing RGB-noise levels.

For each run r in 0..N-1:
  - noise_std = r * (max_std / (N - 1))   (linear sweep; level 0 = baseline)
  - subprocesses scripts/eval_corruption.py inside MolmoBot's venv
  - eval output goes to <out>/run_<R>_noise_<level>/

The eval target (benchmark dir + episode index) is held fixed across runs so
the only thing that changes is the corruption applied to what the policy sees.

Default episode: index 734 of FrankaPickDroidMiniBench (house_770, "pick up the
shaker"). The molmobot config and checkpoint paths mirror run_failure_cases.py.

Usage:
    python scripts/run_corruption_sweep.py \\
        --num_runs 10 \\
        --max_noise_std 0.30 \\
        --out_dir /scratch/gpfs/AM43/yy4041/aim/outputs/house_770_molmobot_noise
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/scratch/gpfs/AM43/yy4041/aim/molmospaces")
MOLMOBOT_DIR = Path("/scratch/gpfs/AM43/yy4041/aim/MolmoBot/MolmoBot")
MOLMOBOT_VENV_PY = MOLMOBOT_DIR / ".venv" / "bin" / "python"
MOLMOBOT_CKPT = Path("/scratch/gpfs/AM43/yy4041/.cache/molmobot/MolmoBot-DROID")
DEFAULT_BENCH_DIR = Path(
    "/scratch/gpfs/AM43/yy4041/.cache/molmospaces/assets/"
    "L3NjcmF0Y2gvZ3Bmcy9BTTQzL3l5NDA0MS9haW0vbW9sbW9zcGFjZXM/"
    "benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/"
    "FrankaPickDroidMiniBench_json_benchmark_20251231"
)
EVAL_CONFIG = "olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig"


def levels_for(num_runs: int, max_std: float) -> list[float]:
    if num_runs == 1:
        return [max_std]
    step = max_std / (num_runs - 1)
    return [round(i * step, 4) for i in range(num_runs)]


def build_env() -> dict:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["JAX_PLATFORMS"] = "cpu"
    env["HF_HUB_OFFLINE"] = "1"
    env["WANDB_MODE"] = "offline"
    env["WANDB_CONSOLE"] = "off"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_runs", type=int, default=10)
    ap.add_argument(
        "--max_noise_std",
        type=float,
        default=0.30,
        help="Max gaussian noise std (fraction of [0,255]). Run r gets r/(N-1) * max.",
    )
    ap.add_argument("--idx", type=int, default=734, help="Episode index in --bench_dir.")
    ap.add_argument("--bench_dir", type=Path, default=DEFAULT_BENCH_DIR)
    ap.add_argument("--task_horizon_steps", type=int, default=600)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument(
        "--cell_timeout_sec",
        type=float,
        default=900.0,
        help="Per-run timeout. molmobot rollouts can be slow.",
    )
    args = ap.parse_args()

    if not MOLMOBOT_VENV_PY.exists():
        sys.exit(f"missing MolmoBot venv: {MOLMOBOT_VENV_PY}")
    if not (MOLMOBOT_CKPT / "model.pt").exists():
        sys.exit(f"missing molmobot checkpoint: {MOLMOBOT_CKPT}/model.pt")
    if not args.bench_dir.exists():
        sys.exit(f"missing bench dir: {args.bench_dir}")

    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    levels = levels_for(args.num_runs, args.max_noise_std)
    print(f"[sweep] {args.num_runs} runs, noise stds: {levels}", flush=True)
    print(f"[sweep] out: {out_root}", flush=True)

    eval_corruption = REPO / "scripts" / "eval_corruption.py"

    failures: list[tuple[int, float, str]] = []
    for r, std in enumerate(levels):
        tag = f"run_{r:02d}_noise_{std:.4f}"
        run_dir = out_root / tag
        log_path = log_dir / f"{tag}.log"
        cmd = [
            str(MOLMOBOT_VENV_PY),
            "-X",
            "faulthandler",
            str(eval_corruption),
            "--noise_std",
            str(std),
            "--noise_seed",
            str(r),
            EVAL_CONFIG,
            "--benchmark_dir",
            str(args.bench_dir),
            "--task_horizon_steps",
            str(args.task_horizon_steps),
            "--idx",
            str(args.idx),
            "--output_dir",
            str(run_dir),
            "--no_wandb",
            "--checkpoint_path",
            str(MOLMOBOT_CKPT),
        ]
        print(f"\n--- {tag} ---", flush=True)
        print("[sweep] " + " ".join(shlex.quote(a) for a in cmd), flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as fh:
            fh.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} | {tag} ===\n")
            fh.flush()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(MOLMOBOT_DIR),
                    env=build_env(),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    timeout=args.cell_timeout_sec,
                )
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                fh.write(f"\n!!! timeout after {int(args.cell_timeout_sec)}s !!!\n")
                rc = 124
        if rc != 0:
            print(f"[sweep] {tag} FAILED (rc={rc}); see {log_path}", flush=True)
            failures.append((r, std, f"rc={rc}"))
        else:
            print(f"[sweep] {tag} done", flush=True)

    print(f"\n[sweep] complete. {args.num_runs - len(failures)}/{args.num_runs} succeeded.",
          flush=True)
    if failures:
        print("[sweep] failures:")
        for r, std, why in failures:
            print(f"  - run_{r:02d} noise={std} ({why})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
