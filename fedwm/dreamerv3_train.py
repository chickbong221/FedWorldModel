#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fedwm.dreamerv3_adapter import DreamerV3Adapter, aggregate_params


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _init_wandb(args: argparse.Namespace, fl_dict: Dict[str, Any]):
    if not args.wandb:
        return None
    import wandb

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    run_name = args.wandb_name or Path(args.logdir).name
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
        config={
            "task": args.task,
            "configs": args.configs,
            "seed": args.seed,
            "logdir": args.logdir,
            "target_env_frames": args.target_env_frames,
            "fed": fl_dict,
            "runner": "fedavg_dreamerv3",
        },
    )


def _split_int(total: int, parts: int) -> List[int]:
    if parts <= 0:
        return []
    base, rem = divmod(max(int(total), 0), int(parts))
    return [base + (1 if idx < rem else 0) for idx in range(parts)]


class DreamerServerFedAvg:
    def __init__(self, cfg: Any, adapter: DreamerV3Adapter):
        self.cfg = cfg
        self.adapter = adapter
        self.num_clients = int(getattr(cfg.fl, "num_clients", 1))
        self.join_ratio = float(getattr(cfg.fl, "join_ratio", 1.0))
        self.random_join_ratio = bool(getattr(cfg.fl, "random_join_ratio", False))
        self.client_drop_rate = float(getattr(cfg.fl, "client_drop_rate", 0.0))
        self.num_join_clients = max(1, int(self.num_clients * self.join_ratio))
        self.updates: List[Dict[str, Any]] = []
        self.last_agg_stats: Dict[str, float] = {}

    def select_clients(self, clients: List[Any]) -> List[Any]:
        if self.random_join_ratio:
            joined = random.choice(range(self.num_join_clients, self.num_clients + 1))
        else:
            joined = self.num_join_clients
        return random.sample(clients, k=joined)

    def reset_round_buffers(self) -> None:
        self.updates = []
        self.last_agg_stats = {}

    def get_payload(self):
        return self.adapter.get_payload()

    def receive_update(self, update: Dict[str, Any]) -> None:
        self.updates.append(update)

    def aggregate(self) -> None:
        active = self.updates
        if self.client_drop_rate > 0 and len(active) > 1:
            keep = max(1, int((1.0 - self.client_drop_rate) * len(active)))
            active = random.sample(active, k=keep)
        state_dict = aggregate_params(active)
        self.adapter.set_payload(state_dict)
        self.last_agg_stats = {
            "num_updates": float(len(active)),
            "total_samples": float(sum(max(int(x.get("num_samples", 1)), 1) for x in active)),
        }

    def log_round(self, round_idx: int) -> None:
        stats = ", ".join(f"{k}={v:.3f}" for k, v in self.last_agg_stats.items())
        print(f"[DreamerV3 FedAvg Server] round={round_idx} {stats}".strip())


class DreamerClientFedAvg:
    def __init__(self, cid: int, cfg: Any, adapter: DreamerV3Adapter):
        self.id = int(cid)
        self.cfg = cfg
        self.adapter = adapter
        self.global_state = None

    def set_client(self, state_dict, reset_opt: bool = False) -> None:
        self.global_state = state_dict
        self.adapter.set_payload(state_dict, reset_opt=reset_opt)

    def local_round(self, collect_env_steps: Optional[int] = None) -> Dict[str, Any]:
        if self.global_state is None:
            raise RuntimeError(f"Client {self.id}: global model not set.")
        steps = int(getattr(self.cfg.fl, "local_steps", 0))
        if collect_env_steps is None:
            print(f"[DreamerV3 Client {self.id}] local_steps={steps}")
        else:
            frames = collect_env_steps * self.adapter.action_repeat()
            print(
                f"[DreamerV3 Client {self.id}] local_steps={steps}, "
                f"collect_env_steps={collect_env_steps}, collect_frames={frames}"
            )
        env_steps_before = self.adapter.env_steps
        metrics = self.adapter.local_train(steps, collect_env_steps=collect_env_steps)
        env_steps_delta = self.adapter.env_steps - env_steps_before
        state_dict = self.adapter.get_payload()
        return {
            "cid": self.id,
            "num_samples": self.adapter.num_samples(),
            "state_dict": state_dict,
            "metrics": metrics,
            "episode_scores": self.adapter.consume_episode_scores(),
            "env_steps": env_steps_delta,
            "env_frames": env_steps_delta * self.adapter.action_repeat(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FedAvg scenario for the DreamerV3 baseline."
    )
    parser.add_argument("--cfp", default=str(_repo_root() / "fedwm" / "config.json"))
    parser.add_argument("--override", default=None, help="JSON override for fedwm/config.json")
    parser.add_argument("--task", default="atari100k_alien")
    parser.add_argument("--configs", nargs="+", default=["atari100k"])
    parser.add_argument("--logdir", default="callbacks/dreamerv3-fedwm/atari100k-alien")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="FedWorldModel-wm_core_baselines_dreamerv3")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_mode", default=None, choices=[None, "online", "offline", "disabled"])
    parser.add_argument(
        "--target_env_frames",
        type=int,
        default=400_000,
        help="Total Atari frames to distribute over all federated rounds. "
        "Use 0 to keep the old update-based collection behavior.",
    )
    args, dreamer_overrides = parser.parse_known_args()

    tmpdir = _repo_root() / ".tmp" / "jax_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmpdir))
    os.environ.setdefault("TEMP", str(tmpdir))
    os.environ.setdefault("TMP", str(tmpdir))

    fl_dict = _load_json(args.cfp)
    if args.override:
        _deep_update(fl_dict, json.loads(args.override))
    cfg = SimpleNamespace(fl=SimpleNamespace(**fl_dict))
    wandb_run = _init_wandb(args, fl_dict)

    global_rounds = int(getattr(cfg.fl, "global_rounds", 1))
    num_clients = int(getattr(cfg.fl, "num_clients", 1))
    actual_env_frames = 0
    latest_client_scores: Dict[int, float] = {}

    server = DreamerServerFedAvg(
        cfg,
        DreamerV3Adapter.from_config(
            configs=args.configs,
            task=args.task,
            logdir=f"{args.logdir}/server",
            seed=args.seed,
            role="server",
            argv_overrides=dreamer_overrides,
        ),
    )
    clients = [
        DreamerClientFedAvg(
            cid,
            cfg,
            DreamerV3Adapter.from_config(
                configs=args.configs,
                task=args.task,
                logdir=f"{args.logdir}/client{cid}",
                seed=args.seed + 1000 * cid,
                role=f"client{cid}",
                argv_overrides=dreamer_overrides,
            ),
        )
        for cid in range(num_clients)
    ]

    for round_idx in range(global_rounds):
        print(f"\n===== DreamerV3 Round {round_idx + 1}/{global_rounds} =====")
        server.reset_round_buffers()
        state_dict = server.get_payload()
        selected = server.select_clients(clients)
        round_updates: List[Dict[str, Any]] = []
        scheduled_env_frames = None
        client_env_steps = [None] * len(selected)
        if args.target_env_frames > 0:
            prev_frames = int(round(args.target_env_frames * round_idx / global_rounds))
            scheduled_env_frames = int(round(args.target_env_frames * (round_idx + 1) / global_rounds))
            round_frame_budget = max(scheduled_env_frames - prev_frames, 0)
            repeat = selected[0].adapter.action_repeat() if selected else 1
            round_agent_steps = round_frame_budget // repeat
            client_env_steps = _split_int(round_agent_steps, len(selected))

        for client, collect_steps in zip(selected, client_env_steps):
            client.set_client(state_dict, reset_opt=False)
            update = client.local_round(collect_env_steps=collect_steps)
            round_updates.append(update)
            server.receive_update(update)
        server.aggregate()
        server.log_round(round_idx)
        actual_env_frames += int(sum(update.get("env_frames", 0) for update in round_updates))
        wandb_step = scheduled_env_frames if scheduled_env_frames is not None else actual_env_frames

        if wandb_run:
            log_data: Dict[str, Any] = {
                "fed/round": round_idx,
                "fed/env_frames": wandb_step,
                "fed/actual_env_frames": actual_env_frames,
                **{f"fed/{k}": v for k, v in server.last_agg_stats.items()},
            }
            for update in round_updates:
                cid = int(update["cid"])
                scores = [float(x) for x in update.get("episode_scores", [])]
                if scores:
                    latest_client_scores[cid] = sum(scores) / len(scores)
                    log_data[f"episode/client_{cid}_score"] = latest_client_scores[cid]
                for key, value in update.get("metrics", {}).items():
                    log_data[f"clients/{cid}/train/{key}"] = value
            if len(latest_client_scores) == num_clients:
                mean_score = sum(latest_client_scores.values()) / len(latest_client_scores)
                log_data["fed/episode_score_mean"] = mean_score
                log_data["episode/score"] = mean_score
            wandb_run.log(log_data, step=wandb_step)

    for client in clients:
        client.adapter.close()
    server.adapter.close()
    if wandb_run:
        wandb_run.finish()
    print("\nDreamerV3 FedAvg scenario done.")


if __name__ == "__main__":
    main()
