# fedwm/server.py
from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch


class ServerWMAvg:
    def __init__(self, cfg: Any, tw: Any):
        self.cfg = cfg
        self.tw = tw  # TwisterAdapter (server-side)

        self.num_clients = int(getattr(cfg.fl, "num_clients", 0))
        self.join_ratio = float(getattr(cfg.fl, "join_ratio", 1.0))
        self.random_join_ratio = bool(getattr(cfg.fl, "random_join_ratio", False))
        self.client_drop_rate = float(getattr(cfg.fl, "client_drop_rate", 0.0))

        self.num_join_clients = max(1, int(self.num_clients * self.join_ratio))
        self.current_num_join_clients = self.num_join_clients

        # round state
        self.selected_clients: List[Any] = []
        self._updates: List[Dict[str, Any]] = []

        # simple logging buffers
        self.round_time_cost: List[float] = []
        self.last_agg_stats: Dict[str, float] = {}

    # ----------------------------
    # Client selection / broadcast
    # ----------------------------
    def select_clients(self, clients: List[Any]) -> List[Any]:

        if self.random_join_ratio:
            self.current_num_join_clients = random.choice(range(self.num_join_clients, self.num_clients + 1))
        else:
            self.current_num_join_clients = self.num_join_clients

        self.selected_clients = random.sample(clients, k=self.current_num_join_clients)
        return self.selected_clients

    def get_payload(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Return (wm_sd, actor_sd, critic_sd) to broadcast.
        """
        return self.tw.get_payload()

    # ----------------------------
    # Receive / aggregate
    # ----------------------------
    def reset_round_buffers(self) -> None:
        self._updates = []
        self.last_agg_stats = {}

    def receive_update(self, update: Dict[str, Any]) -> None:
        self._updates.append(update)

    def _apply_client_drop(self) -> List[Dict[str, Any]]:
        if not self._updates:
            return []

        if self.client_drop_rate <= 0.0:
            return self._updates

        k = max(1, int((1.0 - self.client_drop_rate) * len(self._updates)))
        return random.sample(self._updates, k=k)

    @torch.no_grad()
    def aggregate_world_model(self) -> Dict[str, float]:
        active = self._apply_client_drop()
        if len(active) == 0:
            raise RuntimeError("No client updates received for aggregation.")

        # weights
        weights: List[float] = []
        for upd in active:
            ns = int(upd.get("num_samples", 1))
            weights.append(max(ns, 1))
        denom = float(sum(weights))
        weights = [w / denom for w in weights]

        # init accumulator with first client's keys (assume all match)
        first_sd: Dict[str, torch.Tensor] = active[0]["wm_state_dict"]
        agg: Dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in first_sd.items()}

        # accumulate weighted average
        for w, upd in zip(weights, active):
            sd: Dict[str, torch.Tensor] = upd["wm_state_dict"]
            for k, v in sd.items():
                agg[k] += v.to(dtype=agg[k].dtype) * float(w)

        # load back into server WM
        self.tw.set_payload(
        wm_sd=agg,
        actor_sd={},   # keep existing
        critic_sd={},  # keep existing
        strict=False,
        reset_opt=False,
    )

        # basic stats
        self.last_agg_stats = {
            "num_updates": float(len(active)),
            "total_samples": float(denom),
        }
        return dict(self.last_agg_stats)

    # ----------------------------
    # Server-side actor/critic updates
    # ----------------------------
    def train_actor_critic(self) -> Dict[str, float]:

        n = int(getattr(self.cfg.fl, "server_ac_updates", 0))
        if n <= 0:
            return {}

        out = self.tw.local_wm_train(n, wm_config=self.cfg.wm)

        return _to_float_dict(out)

    # ----------------------------
    # Logging
    # ----------------------------
    def log_round(self, r: int, extra: Optional[Dict[str, float]] = None) -> None:
        """
        Minimal logger. You can later hook wandb/tensorboard.
        """
        msg = {
            "round": float(r),
            **(self.last_agg_stats or {}),
            **(extra or {}),
        }
        # keep it simple for now
        keys = ", ".join([f"{k}={v:.3f}" for k, v in msg.items() if k != "round"])
        print(f"[Server] round={r} {keys}".strip())


def _to_float_dict(d: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out
