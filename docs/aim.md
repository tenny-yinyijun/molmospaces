# Running Policy Inference on MolmoSpaces Benchmarks

## Required env vars


| Var                | What it points at                          | Used by      |
| ------------------ | ------------------------------------------ | ------------ |
| `OPENPI_DIR`       | openpi repo root                           | pi0 / pi05   |
| `OPENPI_DATA_HOME` | dir containing `checkpoints/<ckpt_name>/`  | pi0 / pi05   |
| `MOLMOBOT_DIR`     | `MolmoBot/MolmoBot/` repo root             | MolmoBot     |
| `MOLMOBOT_CKPT`    | weights dir (default `~/.cache/molmobot/MolmoBot-DROID`) | MolmoBot |


---

## Step 1 — one-time: this repo

```bash
uv sync
.venv/bin/python -m molmo_spaces.molmo_spaces_constants   # downloads benchmarks + scene assets
```

---

## Step 2 — one-time: per-policy setup

Only do the section(s) for the policy(s) you want.

### 2a. pi0 / pi05

```bash
# clone openpi anywhere; export OPENPI_DIR if not a sibling of molmospaces
git clone https://github.com/omarrayyann/openpi "$OPENPI_DIR"
cd "$OPENPI_DIR" && uv sync

# download checkpoints (~14 GB each; pick what you need)
mkdir -p "$OPENPI_DATA_HOME/checkpoints"
cd "$OPENPI_DATA_HOME/checkpoints"
gsutil cp -r gs://openpi-assets/checkpoints/pi05_droid_jointpos .
gsutil cp -r gs://openpi-assets/checkpoints/pi0_droid_jointpos .
gsutil cp -r gs://openpi-assets/checkpoints/pi0_fast_droid_jointpos .

# install dependencies in molmospaces
bash bash_scripts/run_pi_local.sh --install-deps
```

### 2b. MolmoBot

```bash
# clone MolmoBot anywhere; export MOLMOBOT_DIR if not a sibling of molmospaces
git clone <MolmoBot repo URL> "$(dirname "$MOLMOBOT_DIR")"
cd "$MOLMOBOT_DIR" && uv sync --extra eval

# weights (~18 GB) — defaults to ~/.cache/molmobot/MolmoBot-DROID
mkdir -p "$(dirname "${MOLMOBOT_CKPT:-$HOME/.cache/molmobot/MolmoBot-DROID}")"
huggingface-cli download allenai/MolmoBot-DROID \
    --local-dir "${MOLMOBOT_CKPT:-$HOME/.cache/molmobot/MolmoBot-DROID}"
```

---

## Step 3 — run inference


| Policy   | Command                                              |
| -------- | ---------------------------------------------------- |
| pi05     | `bash bash_scripts/run_pi_local.sh`                  |
| pi0      | `POLICY=pi0 bash bash_scripts/run_pi_local.sh`       |
| pi0_fast | `POLICY=pi0_fast bash bash_scripts/run_pi_local.sh`  |
| MolmoBot | `bash bash_scripts/run_molmobot_local.sh`            |

| Env var        | Default                                          |
| -------------- | ------------------------------------------------ |
| `BENCH_DIR`    | `FrankaPickDroidMiniBench_json_benchmark_20251231` |
| `MAX_EPISODES` | all                                              |
| `TASK_HORIZON` | 500 (pi*), 600 (MolmoBot)                        |

Example: `MAX_EPISODES=10 POLICY=pi0 bash bash_scripts/run_pi_local.sh`.

