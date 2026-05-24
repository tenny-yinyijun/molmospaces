"""Run pi0, pi05, and molmobot on a fixed set of houses N times each, save labeled videos.

For each (policy, house, repeat r):
  1. Build a per-run mini benchmark JSON containing only the chosen house's
     episodes, with each episode's `seed` set to `r * SEED_STRIDE + house_index`
     so that repeats actually differ from one another (json_eval_runner falls
     back to episode_idx when seed is None, so without this every repeat would
     reuse the same simulator seed).
  2. Invoke molmo_spaces.evaluation.eval_main on that mini benchmark and write
     output under <out_dir>/<policy>/house_<H>/run_<R>/.
  3. Run scripts/collect_eval_videos.py on the resulting eval dir to drop
     success/failure-labeled mp4s next to the raw output.

The seven houses come from the user's failure-case picklist
(72, 520, 78, 196, 237, 68, 86). House 196 has zero episodes in
FrankaPickandPlaceDroidMiniBench_20260111 and is skipped with a warning.

This script handles the policy-server lifecycle for pi0/pi05 (one bring-up per
policy, torn down before the next policy starts) and runs molmobot in-process
out of its own venv.

Usage:
    python scripts/run_failure_cases.py --num_repeats 3 --out_dir /path/to/results
    python scripts/run_failure_cases.py -n 1 -o ./fc --policies pi0 molmobot
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO = Path("/scratch/gpfs/AM43/yy4041/aim/molmospaces")
OPENPI_DIR = Path("/scratch/gpfs/AM43/yy4041/aim/openpi")
MOLMOBOT_DIR = Path("/scratch/gpfs/AM43/yy4041/aim/MolmoBot/MolmoBot")
MOLMOBOT_VENV = MOLMOBOT_DIR / ".venv"
MOLMOBOT_CKPT = Path("/scratch/gpfs/AM43/yy4041/.cache/molmobot/MolmoBot-DROID")
OPENPI_DATA_HOME = Path(
    os.environ.get("OPENPI_DATA_HOME", "/scratch/gpfs/AM43/yy4041/.cache/openpi")
)

DEFAULT_HOUSES = [72, 520, 78, 196, 237, 68, 86]

# Large stride so seed for repeat r doesn't collide with another episode's natural seed.
SEED_STRIDE = 100_000


# A second exocentric camera at the robot's right (mirror of exo_camera_1 across
# the xz-plane). Adds a third recorded mp4 per episode without changing what the
# policies see — they hardcode obs["exo_camera_1"] and ignore the rest.
# Mirror math: pos y -> -y; quat (w, x, y, z) -> (w, -x, y, -z). Verified to
# point back at workspace center in scripts/run_failure_cases.py history.
SECOND_EXO_CAM_SPEC = {
    "name": "exo_camera_2",
    "type": "robot_mounted",
    "reference_body_names": ["robot_0/fr3_link0"],
    "camera_offset": [0.10, -0.57, 0.66],
    "lookat_offset": [0.0, 0.0, 0.08],
    "camera_quaternion": [-0.3633, 0.1241, 0.4263, -0.8191],
    "fov": 71.0,
    "record_depth": False,
}


@dataclass
class PolicySpec:
    name: str
    config: str  # "module:ClassName"
    config_name: str  # eval_output/<config_name>/<timestamp>
    task_horizon: int
    venv_python: Path
    cwd: Path
    # pi0/pi05 only: server bring-up info
    ckpt_name: str | None = None
    port: int | None = None


POLICIES: dict[str, PolicySpec] = {
    "pi0": PolicySpec(
        name="pi0",
        config="molmo_spaces.evaluation.configs.evaluation_configs:Pi0PolicyEvalConfig",
        config_name="Pi0PolicyEvalConfig",
        task_horizon=500,
        venv_python=REPO / ".venv" / "bin" / "python",
        cwd=REPO,
        ckpt_name="pi0_droid_jointpos",
        port=8090,
    ),
    "pi05": PolicySpec(
        name="pi05",
        config="molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig",
        config_name="PiPolicyEvalConfig",
        task_horizon=500,
        venv_python=REPO / ".venv" / "bin" / "python",
        cwd=REPO,
        ckpt_name="pi05_droid_jointpos",
        port=8080,
    ),
    "molmobot": PolicySpec(
        name="molmobot",
        config="olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig",
        config_name="FrankaState8ClampAbsPosConfig",
        task_horizon=600,
        venv_python=MOLMOBOT_VENV / "bin" / "python",
        cwd=MOLMOBOT_DIR,
    ),
}


def find_default_bench_dir() -> Path:
    base = Path("/scratch/gpfs/AM43/yy4041/.cache/molmospaces/assets")
    matches = sorted(
        base.glob(
            "*/benchmarks/molmospaces-bench-v1/procthor-10k/"
            "FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_*_json_benchmark"
        )
    )
    if not matches:
        sys.exit("[run] no FrankaPickandPlaceDroidMiniBench json benchmark found")
    return matches[-1]


def load_benchmark(bench_dir: Path) -> tuple[list[dict], dict]:
    with (bench_dir / "benchmark.json").open() as f:
        episodes = json.load(f)
    with (bench_dir / "benchmark_metadata.json").open() as f:
        meta = json.load(f)
    return episodes, meta


def write_subset_benchmark(
    out_dir: Path, episodes: list[dict], src_meta: dict
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(episodes, indent=2))
    meta = dict(src_meta)
    meta["num_episodes"] = len(episodes)
    meta["num_houses"] = len({ep["house_index"] for ep in episodes})
    meta["description"] = (
        f"Subset benchmark generated by run_failure_cases.py "
        f"(houses={sorted({ep['house_index'] for ep in episodes})})"
    )
    (out_dir / "benchmark_metadata.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def find_free_port() -> int:
    """Ask the kernel for an unused TCP port; race-prone but cheap and good enough."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, deadline_sec: float, server_proc: subprocess.Popen) -> None:
    """Wait until the openpi websocket server is genuinely up on `port`.

    We do an HTTP probe and require the response NOT to look like a foreign
    uvicorn/web service squatting on the port — openpi serves websocket only
    and answers HTTP probes with a 426 Upgrade Required, so we look for that.
    """
    import http.client
    end = time.monotonic() + deadline_sec
    last_err: str | None = None
    while time.monotonic() < end:
        if server_proc.poll() is not None:
            raise RuntimeError(f"policy server exited early (code={server_proc.returncode})")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request("GET", "/")
            resp = conn.getresponse()
            status = resp.status
            server_hdr = (resp.getheader("server") or "").lower()
            conn.close()
            if status == 426 or "websocket" in server_hdr or "openpi" in server_hdr:
                return
            # Foreign HTTP server squatting on the port.
            last_err = f"foreign listener on :{port} (HTTP {status}, server={server_hdr!r})"
        except (ConnectionRefusedError, OSError) as e:
            last_err = repr(e)
        time.sleep(2.0)
    raise TimeoutError(
        f"openpi server on port {port} did not come up cleanly in {deadline_sec}s "
        f"(last probe: {last_err})"
    )


@contextmanager
def policy_server(spec: PolicySpec, log_dir: Path):
    """Bring up an openpi policy server for pi0/pi05; no-op for molmobot.

    Picks a free port at runtime (so we don't stomp on other jobs sharing a
    login node) and sets MOLMOSPACES_PI_PORT so the eval-side config picks up
    the same port.
    """
    if spec.ckpt_name is None:  # molmobot path: no server
        yield None
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / f"{spec.name}_server_{time.strftime('%Y%m%d_%H%M%S')}.log"
    ckpt_dir = OPENPI_DATA_HOME / "checkpoints" / spec.ckpt_name
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"openpi checkpoint not found: {ckpt_dir}")

    port = find_free_port()
    spec.port = port  # so build_eval_env / downstream see the runtime port

    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.60"

    cmd = [
        "uv", "run", "scripts/serve_policy.py",
        f"--port={port}", "policy:checkpoint",
        f"--policy.config={spec.ckpt_name}",
        f"--policy.dir={ckpt_dir}",
    ]
    print(f"[run] starting {spec.name} server on :{port} (log: {server_log})", flush=True)
    log_fh = server_log.open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(OPENPI_DIR),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    try:
        wait_for_port(port, deadline_sec=300, server_proc=proc)
        print(f"[run] {spec.name} server up (pid={proc.pid}, port={port})", flush=True)
        yield proc
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            for _ in range(10):
                if proc.poll() is not None:
                    break
                time.sleep(1.0)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        log_fh.close()
        print(f"[run] {spec.name} server torn down", flush=True)


def build_eval_env(spec: PolicySpec) -> dict:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["JAX_PLATFORMS"] = "cpu"
    env["HF_HUB_OFFLINE"] = "1"
    env["WANDB_MODE"] = "offline"
    env["WANDB_CONSOLE"] = "off"
    env["PYTHONUNBUFFERED"] = "1"
    if spec.port is not None:
        # Read by PiPolicyEvalConfig.model_post_init to override the hardcoded port.
        env["MOLMOSPACES_PI_PORT"] = str(spec.port)
    return env


def latest_eval_output(parent: Path, config_name: str) -> Path | None:
    cand = parent / config_name
    if not cand.exists():
        return None
    runs = sorted(p for p in cand.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def cell_has_trajectory(run_dir: Path) -> bool:
    """Return True iff this cell has at least one trajectories_*.h5 with usable data.

    Used both for retry detection (post-eval) and resume (pre-eval skip).
    Empty H5s and crashed-mid-write H5s count as failure.
    """
    if not run_dir.exists():
        return False
    h5_files = list(run_dir.rglob("trajectories_*.h5"))
    if not h5_files:
        return False
    import h5py
    for h5 in h5_files:
        try:
            with h5py.File(h5, "r") as f:
                # Real outputs have one or more "traj_*" top-level groups.
                if any(k.startswith("traj_") for k in f.keys()):
                    return True
        except (OSError, KeyError):
            continue
    return False


def run_eval(
    spec: PolicySpec,
    bench_dir: Path,
    output_dir: Path,
    log_path: Path,
    timeout_sec: float,
) -> int:
    """Run eval_main.py for one cell. Returns 124 on timeout (POSIX convention)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    args = [
        str(spec.venv_python),
        "-X", "faulthandler",
        "-m", "molmo_spaces.evaluation.eval_main",
        spec.config,
        "--benchmark_dir", str(bench_dir),
        "--task_horizon_steps", str(spec.task_horizon),
        "--output_dir", str(output_dir),
        "--no_wandb",
    ]
    if spec.name == "molmobot":
        args += ["--checkpoint_path", str(MOLMOBOT_CKPT)]

    pretty = " ".join(shlex.quote(a) for a in args)
    print(f"[run] {pretty}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Append so retry attempts share one log file with clear separators.
    with log_path.open("a") as fh:
        fh.write(f"\n=== attempt at {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"(timeout={int(timeout_sec)}s) ===\n")
        fh.flush()
        try:
            proc = subprocess.run(
                args,
                cwd=str(spec.cwd),
                env=build_eval_env(spec),
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
            )
            return proc.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\n!!! eval exceeded {int(timeout_sec)}s timeout, killed !!!\n")
            return 124


def run_collect(spec: PolicySpec, eval_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [
        str(spec.venv_python),
        str(REPO / "scripts" / "collect_eval_videos.py"),
        "--eval_dir", str(eval_dir),
        "--out_dir", str(out_dir),
        "--copy",
    ]
    print(f"[run] collect: {' '.join(shlex.quote(a) for a in args)}", flush=True)
    subprocess.run(args, check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--num_repeats", type=int, required=True,
                    help="Number of repeats per (policy, house). Each repeat uses a different seed.")
    ap.add_argument("-o", "--out_dir", type=Path, required=True,
                    help="Top-level output dir; structure: <out>/<policy>/house_<H>/run_<R>/")
    ap.add_argument("--policies", nargs="+", default=["pi0", "pi05", "molmobot"],
                    choices=list(POLICIES.keys()))
    ap.add_argument("--houses", type=int, nargs="+", default=DEFAULT_HOUSES)
    ap.add_argument("--bench_dir", type=Path, default=None,
                    help="Override base benchmark dir. Default: latest FrankaPickandPlaceDroidMiniBench json bench.")
    ap.add_argument("--max_attempts", type=int, default=3,
                    help="Max retry attempts per (policy, house, run) on timeout / empty output.")
    ap.add_argument("--cell_timeout_sec", type=float, default=600.0,
                    help="Per-attempt timeout in seconds (kills hung renderer).")
    ap.add_argument("--no_resume", action="store_true",
                    help="By default, cells that already have a usable trajectories H5 are skipped. "
                         "Pass this to re-run them.")
    ap.add_argument("--add_second_exo_cam", action="store_true",
                    help="Inject a second exocentric camera ('exo_camera_2') mirrored across the "
                         "xz-plane from exo_camera_1. Produces a third mp4 per episode "
                         "(wrist + 2 side views). Policies still only consume exo_camera_1.")
    args = ap.parse_args()

    if args.num_repeats < 1:
        sys.exit("--num_repeats must be >= 1")

    bench_dir = (args.bench_dir or find_default_bench_dir()).resolve()
    print(f"[run] base benchmark : {bench_dir}", flush=True)
    all_episodes, src_meta = load_benchmark(bench_dir)

    eps_by_house: dict[int, list[dict]] = {}
    for idx, ep in enumerate(all_episodes):
        ep = dict(ep)  # don't mutate the loaded list in place
        ep["_orig_index"] = idx  # for traceability only; EpisodeSpec allows extras
        eps_by_house.setdefault(ep["house_index"], []).append(ep)

    valid_houses: list[int] = []
    for h in args.houses:
        if h in eps_by_house and eps_by_house[h]:
            valid_houses.append(h)
        else:
            print(f"[run] WARNING: house {h} has 0 episodes in this benchmark — skipping",
                  flush=True)
    if not valid_houses:
        sys.exit("[run] no valid houses to evaluate")

    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / ".logs"
    bench_cache = out_root / ".tmp_benchmarks"

    # Cell outcome ledger for the final summary.
    skipped: list[str] = []
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []  # (cell_id, reason)

    def cell_id(p: str, h: int, r: int) -> str:
        return f"{p}/house_{h}/run_{r}"

    overall_rc = 0
    for pname in args.policies:
        spec = POLICIES[pname]
        print(f"\n========== policy: {pname} ==========", flush=True)
        if spec.name in ("pi0", "pi05") and not spec.venv_python.exists():
            print(f"[run] missing venv: {spec.venv_python} — skipping {pname}", flush=True)
            for h in valid_houses:
                for r in range(args.num_repeats):
                    failed.append((cell_id(pname, h, r), "missing venv"))
            continue
        if spec.name == "molmobot":
            if not (MOLMOBOT_CKPT / "model.pt").exists():
                print(f"[run] missing molmobot checkpoint: {MOLMOBOT_CKPT}/model.pt — skipping",
                      flush=True)
                for h in valid_houses:
                    for r in range(args.num_repeats):
                        failed.append((cell_id(pname, h, r), "missing molmobot ckpt"))
                continue

        # Pre-pass: figure out which cells still need work for this policy. If
        # everything is already done, skip server bring-up entirely.
        todo: list[tuple[int, int]] = []  # (house, repeat)
        for house in valid_houses:
            for r in range(args.num_repeats):
                run_dir = out_root / pname / f"house_{house}" / f"run_{r}"
                if (not args.no_resume) and cell_has_trajectory(run_dir):
                    print(f"[run] SKIP {cell_id(pname, house, r)} (already has trajectory data)",
                          flush=True)
                    skipped.append(cell_id(pname, house, r))
                    # Make sure labeled videos exist even if we skipped the eval.
                    eval_out = latest_eval_output(run_dir, spec.config_name)
                    if eval_out is not None and not (run_dir / "labeled").exists():
                        run_collect(spec, eval_out, run_dir / "labeled")
                else:
                    todo.append((house, r))
        if not todo:
            print(f"[run] all {pname} cells already complete — skipping server bring-up",
                  flush=True)
            continue

        with policy_server(spec, log_dir / pname):
            for house, r in todo:
                seed = r * SEED_STRIDE + house  # unique per (house, repeat)
                repeat_eps = []
                for ep in eps_by_house[house]:
                    new_ep = dict(ep)
                    new_ep.pop("_orig_index", None)
                    new_ep["seed"] = seed
                    if args.add_second_exo_cam:
                        cams = list(new_ep.get("cameras") or [])
                        if not any(c.get("name") == "exo_camera_2" for c in cams):
                            cams.append(SECOND_EXO_CAM_SPEC)
                        new_ep["cameras"] = cams
                    repeat_eps.append(new_ep)

                bench_subdir = bench_cache / pname / f"house_{house}" / f"run_{r}"
                write_subset_benchmark(bench_subdir, repeat_eps, src_meta)

                run_dir = out_root / pname / f"house_{house}" / f"run_{r}"
                log_path = log_dir / pname / f"house_{house}_run_{r}.log"

                cid = cell_id(pname, house, r)
                cell_done = False
                last_reason = "unknown"
                for attempt in range(1, args.max_attempts + 1):
                    print(f"\n--- {cid} | seed {seed} | episodes {len(repeat_eps)} | "
                          f"attempt {attempt}/{args.max_attempts} ---", flush=True)
                    rc = run_eval(spec, bench_subdir, run_dir, log_path, args.cell_timeout_sec)
                    if rc == 124:
                        last_reason = f"timeout after {int(args.cell_timeout_sec)}s"
                    elif rc != 0:
                        last_reason = f"non-zero exit ({rc})"
                    elif not cell_has_trajectory(run_dir):
                        last_reason = "eval exited 0 but no trajectories saved"
                    else:
                        cell_done = True
                        break
                    print(f"[run] attempt {attempt} failed: {last_reason} (see {log_path})",
                          flush=True)

                if not cell_done:
                    print(f"[run] GIVING UP on {cid} after {args.max_attempts} attempts: "
                          f"{last_reason}", flush=True)
                    failed.append((cid, last_reason))
                    overall_rc = overall_rc or 1
                    continue

                eval_out = latest_eval_output(run_dir, spec.config_name)
                if eval_out is None:
                    print(f"[run] WARNING: cell succeeded but no eval output dir under "
                          f"{run_dir}/{spec.config_name}", flush=True)
                    failed.append((cid, "no eval output dir"))
                    overall_rc = overall_rc or 1
                    continue
                print(f"[run] eval output: {eval_out}", flush=True)
                run_collect(spec, eval_out, run_dir / "labeled")
                succeeded.append(cid)

    print(f"\n[run] done. Results under: {out_root}")
    print(f"[run] summary: {len(succeeded)} succeeded, {len(skipped)} skipped (resume), "
          f"{len(failed)} failed")
    if failed:
        print("[run] failed cells:")
        for cid, reason in failed:
            print(f"        - {cid}  ({reason})")
        print("[run] re-run the same command to retry failed cells (resume is automatic).")
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
