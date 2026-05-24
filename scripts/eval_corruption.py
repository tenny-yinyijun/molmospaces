"""Run eval_main with gaussian noise added to RGB observations.

Monkey-patches `CameraSensor.get_observation` to corrupt the RGB frame the
policy sees, while caching the clean frame. After each episode,
`save_videos_from_raw_observations` is wrapped to additionally write a
`*_clean.mp4` next to each corrupted camera mp4, so you can compare exactly
what the policy saw versus what the simulator rendered.

Usage (must run in a venv where molmo_spaces and the eval config class are
both importable — for molmobot that means MolmoBot/.venv):

    python eval_corruption.py --noise_std 0.1 --noise_seed 0 \\
        olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig \\
        --benchmark_dir <bench> --idx 734 --output_dir <out> \\
        --task_horizon_steps 600 --no_wandb \\
        --checkpoint_path /scratch/gpfs/AM43/yy4041/.cache/molmobot/MolmoBot-DROID

Any flag the script doesn't recognize is forwarded verbatim to eval_main.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument(
        "--noise_std",
        type=float,
        default=None,
        help="Gaussian noise std as a fraction of full RGB range (0..1). "
        "0 disables corruption. Defaults to $MOLMOSPACES_NOISE_STD or 0.0.",
    )
    p.add_argument(
        "--noise_seed",
        type=int,
        default=0,
        help="Seed for the corruption RNG. Independent of the simulator seed.",
    )
    known, passthrough = p.parse_known_args()
    if known.noise_std is None:
        known.noise_std = float(os.environ.get("MOLMOSPACES_NOISE_STD", "0.0"))
    return known, passthrough


def install_patches(noise_std: float, noise_seed: int) -> None:
    from molmo_spaces.env import sensors_cameras
    from molmo_spaces.utils import save_utils

    rng = np.random.default_rng(noise_seed)
    # Per-camera buffer of clean frames. Reset each episode by patched_save.
    clean_frames: dict[str, list[np.ndarray]] = {}

    def patched_get(self, env, task, batch_index: int = 0, *args, **kwargs):
        frame = env.render_rgb_frame(self.camera_name)
        if frame is None:
            w, h = self.img_resolution
            return np.zeros((h, w, 3), dtype=np.uint8)
        clean_frames.setdefault(self.camera_name, []).append(frame.copy())
        if noise_std <= 0.0:
            return frame
        noise = rng.normal(loc=0.0, scale=noise_std * 255.0, size=frame.shape)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    sensors_cameras.CameraSensor.get_observation = patched_get

    orig_save = save_utils.save_videos_from_raw_observations

    def patched_save(
        observations_list,
        save_dir,
        fps,
        episode_idx: int = 0,
        save_file_suffix: str = "",
        sensor_suite=None,
    ) -> None:
        orig_save(
            observations_list,
            save_dir,
            fps,
            episode_idx=episode_idx,
            save_file_suffix=save_file_suffix,
            sensor_suite=sensor_suite,
        )
        os.makedirs(save_dir, exist_ok=True)
        for cam_name, frames in clean_frames.items():
            if not frames:
                continue
            path = os.path.join(
                save_dir,
                f"episode_{episode_idx:08d}_{cam_name}_clean{save_file_suffix}.mp4",
            )
            save_utils.save_frames_to_mp4(np.asarray(frames, dtype=np.uint8), path, fps=fps)
        clean_frames.clear()

    save_utils.save_videos_from_raw_observations = patched_save


def main() -> int:
    known, passthrough = parse_args()
    install_patches(known.noise_std, known.noise_seed)
    sys.argv = ["eval_main.py", *passthrough]
    from molmo_spaces.evaluation import eval_main as em

    em.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
