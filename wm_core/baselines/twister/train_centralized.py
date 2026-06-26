import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch

os.environ.setdefault("MUJOCO_GL", "egl")

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_wm_core_on_path() -> None:
    wm_core = _repo_root() / "wm_core"
    if str(wm_core) not in sys.path:
        sys.path.insert(0, str(wm_core))


def _load_config(config_file: str, env_name: str, override_config: dict):
    os.environ["env_name"] = env_name
    os.environ["override_config"] = json.dumps(override_config)

    module_name = config_file.replace(".py", "").replace("/", ".")
    return importlib.import_module(module_name)


def _model_step_value(model) -> int:
    step = getattr(model, "model_step", 0)
    return int(step.item()) if hasattr(step, "item") else int(step)


def _action_step_value(model) -> int:
    step = getattr(model, "action_step", 0)
    return int(step.item()) if hasattr(step, "item") else int(step)


def _steps_per_action_step(model) -> float:
    cfg = model.config
    num_env_steps = (cfg.batch_size * cfg.L) / (cfg.env_step_period * cfg.num_envs)
    return float(num_env_steps * cfg.env_params.get("action_repeat", 1) * cfg.num_envs)


def _find_checkpoint(callback_path: str, checkpoint: Optional[str], load_last: bool) -> Optional[str]:
    if checkpoint:
        return checkpoint if os.path.isabs(checkpoint) else os.path.join(callback_path, checkpoint)
    if not load_last:
        return None

    import functions

    return functions.find_last_checkpoint(callback_path, return_full_path=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Centralized TWISTER baseline training for FedWorldModel."
    )
    parser.add_argument("--env_name", required=True, help="Example: atari100k-alien")
    parser.add_argument(
        "--config_file",
        default="wm_core/configs/twister.py",
        help="TWISTER Python config, relative to repo root.",
    )
    parser.add_argument(
        "--target_action_steps",
        type=int,
        default=400_000,
        help="Train until model.action_step reaches this value. Set 0 to ignore.",
    )
    parser.add_argument(
        "--target_model_steps",
        type=int,
        default=None,
        help="Train until optimizer/model_step reaches this value. Overrides target_action_steps.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Number of model updates per fit epoch. Defaults to TWISTER epoch_length.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10_000,
        help="Save a checkpoint every N model steps.",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path or filename to resume.")
    parser.add_argument("--load_last", action="store_true", help="Resume from last checkpoint.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--wandb_project", default="FedWorldModel")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--keep_last_k", type=int, default=5)
    parser.add_argument("--step_log_period", type=int, default=100)
    parser.add_argument("--eval_period_epoch", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--verbose_progress_bar", type=int, default=1)
    parser.add_argument(
        "--override_config",
        default="{}",
        help='JSON TWISTER override, e.g. \'{"model_size":"SM"}\'.',
    )
    args = parser.parse_args()

    _ensure_wm_core_on_path()

    override_config = json.loads(args.override_config)
    cfg = _load_config(args.config_file, args.env_name, override_config)

    model = cfg.model
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)

    ckpt = _find_checkpoint(cfg.callback_path, args.checkpoint, args.load_last)
    if ckpt:
        model.load(ckpt)

    import functions

    dataset_train, dataset_eval = functions.load_datasets(
        cfg,
        replay_buffer=cfg.training_dataset,
        eval_dataset=cfg.evaluation_dataset,
    )

    max_steps_per_epoch = args.steps_per_epoch or int(model.config.epoch_length)

    if args.target_model_steps is not None:
        target_model_steps = int(args.target_model_steps)
        target_desc = f"model_step={target_model_steps}"
    else:
        action_target = int(args.target_action_steps)
        if action_target <= 0:
            raise ValueError("Set --target_model_steps or a positive --target_action_steps.")
        per_model_step = _steps_per_action_step(model)
        target_model_steps = math.ceil(action_target / per_model_step)
        target_desc = (
            f"action_step~={action_target} "
            f"(converted to model_step={target_model_steps}, {per_model_step:g} action steps/update)"
        )

    print(f"Centralized baseline: env={args.env_name}, target {target_desc}")
    print(f"callback_path={cfg.callback_path}")

    if not args.no_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed. Run `pip install wandb` or pass `--no_wandb`.")
        run_name = args.wandb_run_name or f"centralized-{args.env_name}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "env_name": args.env_name,
                "target_model_steps": target_model_steps,
                "target_action_steps": args.target_action_steps,
                "steps_per_epoch": max_steps_per_epoch,
                "checkpoint_every": args.checkpoint_every,
                "override_config": override_config,
                "callback_path": cfg.callback_path,
                "mode": "centralized",
            },
        )
        wandb.define_metric("train_step")
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="train_step")

    next_checkpoint_step = (
        (_model_step_value(model) // args.checkpoint_every + 1) * args.checkpoint_every
        if args.checkpoint_every > 0
        else target_model_steps
    )

    while _model_step_value(model) < target_model_steps:
        current = _model_step_value(model)
        remaining = target_model_steps - current
        epoch_updates = min(max_steps_per_epoch, remaining)
        cfg.training_dataset.epoch_length = epoch_updates

        print(
            f"Training chunk: model_step={current}, action_step={_action_step_value(model)}, "
            f"next_updates={epoch_updates}"
        )
        model.fit(
            dataset_train=dataset_train,
            epochs=1,
            dataset_eval=dataset_eval,
            eval_steps=args.eval_steps,
            initial_epoch=0,
            callback_path=cfg.callback_path,
            steps_per_epoch=None,
            precision=getattr(cfg, "precision", torch.float32),
            accumulated_steps=getattr(cfg, "accumulated_steps", 1),
            eval_period_step=None,
            eval_period_epoch=args.eval_period_epoch,
            saving_period_epoch=None,
            log_figure_period_step=None,
            log_figure_period_epoch=None,
            step_log_period=args.step_log_period,
            grad_init_scale=getattr(cfg, "grad_init_scale", 65536.0),
            detect_anomaly=getattr(cfg, "detect_anomaly", False),
            recompute_metrics=getattr(cfg, "recompute_metrics", False),
            wandb_logging=not args.no_wandb,
            verbose_progress_bar=args.verbose_progress_bar,
            keep_last_k=args.keep_last_k,
        )

        step = _model_step_value(model)
        if step >= next_checkpoint_step or step >= target_model_steps:
            checkpoint_path = os.path.join(cfg.callback_path, f"checkpoints_step_{step}.ckpt")
            model.save(checkpoint_path, keep_last_k=args.keep_last_k)
            if args.checkpoint_every > 0:
                while next_checkpoint_step <= step:
                    next_checkpoint_step += args.checkpoint_every

    print(
        f"Done: model_step={_model_step_value(model)}, "
        f"action_step={_action_step_value(model)}, callback_path={cfg.callback_path}"
    )
    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
