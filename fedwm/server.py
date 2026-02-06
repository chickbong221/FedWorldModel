# fedwm/server.py
from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch


class ServerFedAvgWM:
    """
    FedAvg server for Federated World Models.

    Designed to feel like PFLlib servers (serverbase/serveravg): select -> send -> train -> receive -> aggregate
    :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

    But adapted to your fedwm/main.py contract:
      - select_clients(clients)
      - get_payload()
      - receive_update(update)
      - aggregate_world_model()
      - train_actor_critic()
      - log_round(r)
    """

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
        """
        Mirror PFLlib behavior: random subset of clients each round :contentReference[oaicite:2]{index=2}
        """
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
        # TwisterAdapter provides get_payload(); else build from individual getters.
        if hasattr(self.tw, "get_payload"):
            return self.tw.get_payload()

        wm_sd = self.tw.get_wm_state_dict(to_cpu=True)
        actor_sd = self.tw.get_actor_state_dict(to_cpu=True)
        critic_sd = self.tw.get_critic_state_dict(to_cpu=True)
        return wm_sd, actor_sd, critic_sd

    # ----------------------------
    # Receive / aggregate
    # ----------------------------
    def reset_round_buffers(self) -> None:
        self._updates = []
        self.last_agg_stats = {}

    def receive_update(self, update: Dict[str, Any]) -> None:
        """
        update expected from ClientWMBase.package_update():
          {
            "cid": int,
            "num_samples": int,
            "wm_state_dict": Dict[str, Tensor],
            "metrics": Dict[str, float],
          }
        """
        self._updates.append(update)

    def _apply_client_drop(self) -> List[Dict[str, Any]]:
        """
        Mirror PFLlib's client_drop_rate logic :contentReference[oaicite:3]{index=3}
        """
        if not self._updates:
            return []

        if self.client_drop_rate <= 0.0:
            return self._updates

        k = max(1, int((1.0 - self.client_drop_rate) * len(self._updates)))
        return random.sample(self._updates, k=k)

    @torch.no_grad()
    def aggregate_world_model(self) -> Dict[str, float]:
        """
        FedAvg over WM params only:
          w_global = sum_i (n_i / sum n) * w_i

        Mirrors Server.aggregate_parameters() but on state_dict tensors instead of model.parameters
        :contentReference[oaicite:4]{index=4}
        """
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
        self.tw.set_wm_state_dict(agg, strict=False)

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
        """
        Train actor/critic on server for cfg.fl.server_ac_updates steps.

        Best practice: freeze world model params so only policy/value updates.

        Requires TwisterAdapter to support either:
          - actor_critic_train_steps(n, wm_config=cfg.wm)
        OR (fallback):
          - set_requires_grad_world_model(enabled)
          - set_requires_grad_actor_critic(enabled)
          - local_wm_train(n, wm_config=cfg.wm, freeze_actor_critic=False)
        """
        n = int(getattr(self.cfg.fl, "server_ac_updates", 0))
        if n <= 0:
            return {}

        # Preferred explicit API
        if hasattr(self.tw, "actor_critic_train_steps"):
            out = self.tw.actor_critic_train_steps(n, wm_config=self.cfg.wm)
            return _to_float_dict(out)

        # Fallback: freeze WM modules, run fit loop steps (which will update whatever requires_grad=True)
        self._set_requires_grad_world_model(False)
        self._set_requires_grad_actor_critic(True)

        # use adapter's training primitive
        if hasattr(self.tw, "local_wm_train"):
            out = self.tw.local_wm_train(
                n,
                wm_config=self.cfg.wm,
                freeze_actor_critic=False,  # do NOT freeze AC here
            )
        elif hasattr(self.tw, "wm_train_steps"):
            out = self.tw.wm_train_steps(n, wm_config=self.cfg.wm)
        else:
            raise AttributeError("TwisterAdapter must provide local_wm_train(...) or wm_train_steps(...).")

        # restore grads
        self._set_requires_grad_world_model(True)
        self._set_requires_grad_actor_critic(True)

        return _to_float_dict(out)

    def _set_requires_grad_actor_critic(self, enabled: bool) -> None:
        # Prefer adapter helper if available
        if hasattr(self.tw, "set_requires_grad_actor_critic"):
            self.tw.set_requires_grad_actor_critic(enabled)
            return

        # Fallback: toggle on TWISTER known module names
        for name in ("policy_network", "value_network", "v_target"):
            m = getattr(self.tw.model, name, None)
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad = enabled

    def _set_requires_grad_world_model(self, enabled: bool) -> None:
        # Prefer adapter helper if available
        if hasattr(self.tw, "set_requires_grad_world_model"):
            self.tw.set_requires_grad_world_model(enabled)
            return

        # Fallback: toggle on common TWISTER WM module names
        for name in ("encoder_network", "rssm", "decoder_network", "reward_network", "continue_network"):
            m = getattr(self.tw.model, name, None)
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad = enabled

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
