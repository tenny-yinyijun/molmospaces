"""Collect per-episode eval videos into a flat folder labeled with success/failure.

Reads success status from the trajectory h5 files written by eval_main.py and
copies/symlinks each per-camera mp4 into the output directory with a
``_success`` or ``_failure`` suffix in the filename.

Usage:
    python scripts/collect_eval_videos.py \\
        --eval_dir eval_output/Pi0PolicyEvalConfig/20260506_120000 \\
        --out_dir failure_analysis/pi0
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from molmo_spaces.utils.eval_utils import collect_episode_results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval_dir",
        required=True,
        type=Path,
        help="Eval-output dir (the timestamped one with house_* subdirs and trajectories*.h5).",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        type=Path,
        help="Where to drop the labeled videos (one flat folder).",
    )
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copy videos instead of symlinking (default: symlink).",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = collect_episode_results(args.eval_dir)
    if not results:
        raise SystemExit(f"No episode results found under {args.eval_dir}")

    success_status = {(r.house_id, r.episode_idx): r.success for r in results}

    pattern = re.compile(r"^episode_(\d+)_(.+)$")
    n_written = 0
    n_skipped = 0
    for vid in sorted(args.eval_dir.glob("house_*/episode_*.mp4")):
        m = pattern.match(vid.stem)
        if not m:
            n_skipped += 1
            continue
        episode_idx = int(m.group(1))
        cam_and_batch = m.group(2)
        house_id = vid.parent.name
        ok = success_status.get((house_id, episode_idx))
        if ok is None:
            n_skipped += 1
            continue
        suffix = "success" if ok else "failure"
        new_name = f"{house_id}_episode_{episode_idx:08d}_{cam_and_batch}_{suffix}.mp4"
        target = args.out_dir / new_name
        if target.exists() or target.is_symlink():
            target.unlink()
        if args.copy:
            shutil.copy2(vid, target)
        else:
            target.symlink_to(vid.resolve())
        n_written += 1

    n_success = sum(1 for ok in success_status.values() if ok)
    n_total = len(success_status)
    print(
        f"Wrote {n_written} labeled videos to {args.out_dir} "
        f"({n_skipped} skipped). Episodes: {n_success}/{n_total} success "
        f"({100 * n_success / max(n_total, 1):.1f}%)."
    )


if __name__ == "__main__":
    main()
