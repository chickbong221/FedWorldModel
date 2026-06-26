# DreamerV3 Baseline

This directory contains the Danijar Hafner DreamerV3 baseline in an isolated
layout under `wm_core/baselines/`.

- `vendor/` is a trimmed copy of upstream `danijar/dreamerv3` runtime code for
  Atari baselines.
- `run_dreamerv3.py` is the local wrapper used from this repo.
- `check_env.py` verifies the Python/JAX/CUDA/runtime dependencies.

Upstream commit:

```text
e3f02248693a79dc8b0ebd62c93683888ddaccfe
```

## Environment

The current upstream DreamerV3 requires Python 3.11+ and JAX 0.4.33. It cannot
be installed into the `fedwm` environment, which is Python 3.8 for TWISTER.
Use the existing `dreamer` environment for this baseline:

```bash
conda activate dreamer
pip install -r wm_core/baselines/requirements.txt
```

Check CUDA/JAX:

```bash
python wm_core/baselines/dreamerv3/check_env.py
```

## Run Atari100k Alien

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python wm_core/baselines/dreamerv3/run_dreamerv3.py \
  --task atari100k_alien \
  --steps 1000 \
  --configs atari100k debug \
  --logdir callbacks/dreamerv3/atari100k-alien-smoke
```

Training baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python wm_core/baselines/dreamerv3/run_dreamerv3.py \
  --task atari100k_alien \
  --steps 100000 \
  --configs atari100k size12m \
  --logdir callbacks/dreamerv3/atari100k-alien
```

To enable W&B logging, add:

```bash
--wandb
```

## FedWM Scenario

DreamerV3 Fed training lives under `fedwm/` and uses a basic FedAvg scenario:

1. Server broadcasts the full DreamerV3 parameters.
2. Clients collect Atari data and train the full DreamerV3 agent locally.
3. Clients send their full updated parameters back to the server.
4. Server aggregates client parameters with FedAvg and does not train locally.

Run a small smoke test:

```bash
conda activate dreamer
CUDA_VISIBLE_DEVICES=0 python fedwm/dreamerv3_train.py \
  --configs atari100k debug \
  --override '{"num_clients":1,"global_rounds":1,"local_steps":1}' \
  --run.envs 1 \
  --run.train_ratio 1 \
  --batch_size 2 \
  --batch_length 8 \
  --replay.size 1000 \
  --jax.prealloc False
```
