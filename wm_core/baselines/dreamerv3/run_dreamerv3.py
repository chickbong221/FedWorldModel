#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
REPO_ROOT = ROOT.parents[2]


def _add_vendor_path() -> None:
    vendor = str(VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run the isolated Danijar DreamerV3 baseline."
    )
    parser.add_argument("--task", default="atari100k_alien")
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["atari100k"],
        help="DreamerV3 config blocks, for example: atari100k size12m",
    )
    parser.add_argument(
        "--logdir",
        default="callbacks/dreamerv3/atari100k-alien",
        help="Output directory for checkpoints, replay, and metrics.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--jax-platform",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Use cuda for real runs; cpu is only for dependency debugging.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, overrides = _parse_args()
    _add_vendor_path()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    tmpdir = REPO_ROOT / ".tmp" / "jax_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmpdir))
    os.environ.setdefault("TEMP", str(tmpdir))
    os.environ.setdefault("TMP", str(tmpdir))

    argv = [
        "--configs",
        *args.configs,
        "--task",
        args.task,
        "--run.steps",
        str(args.steps),
        "--logdir",
        args.logdir,
        "--seed",
        str(args.seed),
        "--jax.platform",
        args.jax_platform,
    ]
    if args.wandb:
        argv.extend(["--logger.outputs", "jsonl", "scope", "wandb"])
    argv.extend(overrides)

    from dreamerv3.main import main as dreamer_main

    dreamer_main(argv)


if __name__ == "__main__":
    main()
