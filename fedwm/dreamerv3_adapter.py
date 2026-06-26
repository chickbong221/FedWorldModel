from __future__ import annotations

import copy
import os
import random
import sys
import collections
from functools import partial as bind
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR = REPO_ROOT / "wm_core" / "baselines" / "dreamerv3" / "vendor"


def ensure_vendor_path() -> None:
    vendor = str(VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def load_dreamer_config(
    *,
    configs: Iterable[str],
    task: str,
    logdir: str,
    seed: int,
    argv_overrides: Optional[List[str]] = None,
    role: str = "client",
):
    ensure_vendor_path()

    import elements
    import ruamel.yaml as yaml

    cfg_file = VENDOR / "dreamerv3" / "configs.yaml"
    loaded = yaml.YAML(typ="safe").load(elements.Path(cfg_file).read())

    cfg = elements.Config(loaded["defaults"])
    for name in configs:
        cfg = cfg.update(loaded[name])

    cfg = cfg.update(
        task=task,
        logdir=logdir,
        seed=int(seed),
        jax={"platform": "cuda", "prealloc": False},
    )

    if argv_overrides:
        cfg = elements.Flags(cfg).parse(argv_overrides)
    return cfg


class DreamerV3Adapter:
    """Small FedAvg adapter around the vendored DreamerV3 JAX agent."""

    WM_PREFIXES = ("enc/", "dyn/", "dec/", "rew/", "con/")
    ACTOR_PREFIXES = ("pol/",)
    CRITIC_PREFIXES = ("val/", "slowval/", "retnorm/", "valnorm/", "advnorm/")

    def __init__(self, config: Any, role: str):
        ensure_vendor_path()

        import embodied
        from dreamerv3.main import make_agent, make_env, make_replay, make_stream

        self.config = config
        self.role = str(role)
        self.agent = make_agent(config)
        self.agent.jaxcfg.profiler = False
        self.replay = make_replay(config, f"replay_{self.role}")
        self.stream_train = iter(self.agent.stream(make_stream(config, self.replay, "train")))
        self.carry_train = self.agent.init_train(config.batch_size)
        self.env_steps = 0
        self._episode_scores = collections.defaultdict(float)
        self._recent_episode_scores: List[float] = []

        fns = [bind(make_env, config, i) for i in range(int(config.run.envs))]
        self.driver = embodied.Driver(fns, parallel=not bool(config.run.debug))
        self.driver.on_step(lambda tran, worker: self._count_step())
        self.driver.on_step(self._record_episode_score)
        self.driver.on_step(self.replay.add)
        self.driver.reset(self.agent.init_policy)

        self._state = self.agent.save()

    @classmethod
    def from_config(
        cls,
        *,
        configs: Iterable[str],
        task: str,
        logdir: str,
        seed: int,
        role: str,
        argv_overrides: Optional[List[str]] = None,
    ) -> "DreamerV3Adapter":
        cfg = load_dreamer_config(
            configs=configs,
            task=task,
            logdir=logdir,
            seed=seed,
            argv_overrides=argv_overrides,
            role=role,
        )
        return cls(cfg, role=role)

    def _count_step(self) -> None:
        self.env_steps += 1

    def _record_episode_score(self, tran: Dict[str, Any], worker: int) -> None:
        if bool(np.asarray(tran["is_first"])):
            self._episode_scores[worker] = 0.0
        self._episode_scores[worker] += float(np.asarray(tran["reward"]))
        if bool(np.asarray(tran["is_last"])):
            self._recent_episode_scores.append(float(self._episode_scores[worker]))
            self._episode_scores[worker] = 0.0

    def lease_to(self, device: Any = None) -> None:
        # JAX placement is controlled by CUDA_VISIBLE_DEVICES and config.jax.
        return None

    def sync_to_cpu(self, free_cuda: bool = True) -> None:
        self._state = self.agent.save()

    def close(self) -> None:
        for env in getattr(self.driver, "envs", []):
            try:
                env.close()
            except Exception:
                pass

    def _split_params(self, params: Optional[Dict[str, np.ndarray]] = None):
        params = params or self._state["params"]
        wm, actor, critic = {}, {}, {}
        for key, value in params.items():
            if key.startswith(self.WM_PREFIXES):
                wm[key] = value
            elif key.startswith(self.ACTOR_PREFIXES):
                actor[key] = value
            elif key.startswith(self.CRITIC_PREFIXES):
                critic[key] = value
        return wm, actor, critic

    def get_payload(self, to_cpu: bool = True):
        return self.get_state_dict(to_cpu=to_cpu)

    def set_payload(
        self,
        state_dict: Dict[str, np.ndarray],
        actor_sd: Optional[Dict[str, np.ndarray]] = None,
        critic_sd: Optional[Dict[str, np.ndarray]] = None,
        *,
        reset_opt: bool = False,
    ) -> None:
        if actor_sd or critic_sd:
            merged = dict(state_dict or {})
            merged.update(actor_sd or {})
            merged.update(critic_sd or {})
            state_dict = merged
        self.set_state_dict(state_dict, reset_opt=reset_opt)

    def get_state_dict(self, to_cpu: bool = True) -> Dict[str, np.ndarray]:
        self._state = self.agent.save()
        return copy.deepcopy(self._state["params"])

    def set_state_dict(
        self,
        state_dict: Dict[str, np.ndarray],
        *,
        reset_opt: bool = False,
    ) -> None:
        state = copy.deepcopy(self._state)
        params = dict(state["params"])
        for key, value in (state_dict or {}).items():
            params[key] = np.asarray(value)

        if reset_opt:
            params = {k: v for k, v in params.items() if not k.startswith("opt/")}
            current = self.agent.save()["params"]
            params.update({k: v for k, v in current.items() if k.startswith("opt/")})

        state["params"] = params
        self.agent.load(state)
        self._state = self.agent.save()

    def local_train(self, steps: int, collect_env_steps: Optional[int] = None) -> Dict[str, float]:
        steps = max(int(steps), 0)
        collect_env_steps = None if collect_env_steps is None else max(int(collect_env_steps), 0)
        if steps == 0 and not collect_env_steps:
            return {}

        min_replay = int(self.config.batch_size) * int(self.config.batch_length)
        policy = lambda *args: self.agent.policy(*args, mode="train")
        collect_chunk = int(getattr(self.config.run, "fedwm_collect_chunk", 10))
        env_steps_per_update = int(getattr(self.config.run, "fedwm_env_steps_per_update", 10))
        metrics: Dict[str, float] = {}

        if collect_env_steps is None:
            while len(self.replay) < min_replay:
                self.driver(policy, steps=collect_chunk)
        elif collect_env_steps > 0:
            remaining = collect_env_steps
            while remaining > 0:
                chunk = min(collect_chunk, remaining)
                self.driver(policy, steps=chunk)
                remaining -= chunk

        if len(self.replay) < min_replay:
            self._state = self.agent.save()
            return metrics

        for _ in range(steps):
            if collect_env_steps is None and env_steps_per_update > 0:
                self.driver(policy, steps=env_steps_per_update)
            batch = next(self.stream_train)
            self.carry_train, outs, mets = self.agent.train(self.carry_train, batch)
            if "replay" in outs:
                self.replay.update(outs["replay"])
            metrics.update(_float_metrics(mets))

        self._state = self.agent.save()
        return metrics

    def local_wm_train(self, steps: int, wm_config: Any = None) -> Dict[str, float]:
        return self.local_train(steps)

    def num_samples(self) -> int:
        return max(int(len(self.replay)), 1)

    def consume_episode_scores(self) -> List[float]:
        scores = list(self._recent_episode_scores)
        self._recent_episode_scores.clear()
        return scores

    def action_repeat(self) -> int:
        task_prefix = str(self.config.task).split("_")[0]
        env_cfg = self.config.env.get(task_prefix, {})
        return int(env_cfg.get("repeat", 1))


def aggregate_params(updates: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not updates:
        raise RuntimeError("No DreamerV3 client updates to aggregate.")

    weights = [max(int(upd.get("num_samples", 1)), 1) for upd in updates]
    denom = float(sum(weights))
    weights = [w / denom for w in weights]

    first = updates[0]["state_dict"]
    agg: Dict[str, np.ndarray] = {}
    for key, value in first.items():
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.floating):
            out = np.zeros_like(arr)
            for weight, update in zip(weights, updates):
                out += np.asarray(update["state_dict"][key], dtype=arr.dtype) * weight
            agg[key] = out
        else:
            agg[key] = arr.copy()
    return agg


def _float_metrics(metrics: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    for key, value in metrics.items():
        try:
            arr = np.asarray(value)
            if arr.size == 1:
                out[key] = float(arr.reshape(()))
        except Exception:
            pass
    return out
