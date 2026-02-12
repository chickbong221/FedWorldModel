# fedwm/client.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Type

import torch


class ClientWMBase:

    def __init__(self, cfg: Any, cid: int, tw: Any, **kwargs):
        self.cfg = cfg
        self.id = int(cid)
        self.tw = tw  # TwisterAdapter (per-client instance)

        # bookkeeping (similar spirit to PFLlib)
        self.train_time_cost = {"num_rounds": 0, "total_cost": 0.0}
        self.send_time_cost = {"num_rounds": 0, "total_cost": 0.0}

        # cached server payloads
        self._global_wm_sd: Optional[Dict[str, torch.Tensor]] = None
        self._actor_sd: Optional[Dict[str, torch.Tensor]] = None
        self._critic_sd: Optional[Dict[str, torch.Tensor]] = None

    # ---------- payload receive ----------
    def set_client(
        self,
        wm_state_dict: Dict[str, torch.Tensor],
        actor_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        critic_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        *,
        reset_opt: bool = False,
    ) -> None:
        # store (optional, for debugging/logging)
        self._global_wm_sd = wm_state_dict
        self._actor_sd = actor_state_dict
        self._critic_sd = critic_state_dict

        # If actor/critic are omitted, keep local ones by sending empty dicts
        actor_state_dict = actor_state_dict or {}
        critic_state_dict = critic_state_dict or {}

        # Apply in one shot (updates adapter CPU snapshot, does NOT overwrite opt/scaler)
        self.tw.set_payload(
            wm_sd=wm_state_dict,
            actor_sd=actor_state_dict,
            critic_sd=critic_state_dict,
            strict=False,
            reset_opt=reset_opt,
        )

    # ---------- algorithm hooks ----------
    def local_initialization(self, received_global_model: Dict[str, torch.Tensor]) -> None:
        """
        Hook for algorithms like ALA that do a special local init *before* training,
        e.g. adaptive local aggregation. PFLlib clientALA uses this hook
        Default: do nothing.
        """
        return

    def train(self) -> Dict[str, float]:
        raise NotImplementedError

    # ---------- update packaging ----------
    def _num_samples_for_fedavg(self) -> int:
        """
        FedAvg weight.
        For world models, a decent proxy is replay buffer experience count.
        Your ReplayBuffer tracks num_steps.
        """
        rb = getattr(self.tw, "replay_buffer", None)
        if rb is None:
            return 1
        ns = getattr(rb, "num_steps", None)
        if ns is None:
            return 1
        try:
            v = int(ns.item()) if hasattr(ns, "item") else int(ns)
            return max(v, 1)
        except Exception:
            return 1

    def package_update(self, metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if metrics is None:
            metrics = {}

        # Always derive from payload (works for both old/new adapter shapes)
        wm_sd, _, _ = self.tw.get_payload(to_cpu=True)

        return {
            "cid": self.id,
            "num_samples": self._num_samples_for_fedavg(),
            "wm_state_dict": wm_sd,
            "metrics": metrics,
        }

    def local_round(self) -> Dict[str, Any]:
        if self._global_wm_sd is None:
            raise RuntimeError(
                f"Client {self.id}: global WM not set."
            )

        start = time.time()

        # Hook to mirror PFLlib (e.g., clientALA.local_initialization)
        self.local_initialization(self._global_wm_sd)

        metrics = self.train() or {}

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start

        return self.package_update(metrics=metrics)


class ClientWMAvg(ClientWMBase):
    """
    FedAvg-style client: run local WM steps then return weights.
    Mirrors PFLlib's clientAVG where train() is the only override.
    """

    def train(self) -> Dict[str, float]:
        # print("training client", self.id)
        local_steps = int(getattr(self.cfg.fl, "local_steps", 0))
        if local_steps <= 0:
            return {}

        print("training client", self.id)
        out = self.tw.local_wm_train(local_steps, wm_config=self.cfg.wm)

        # normalize metrics to float dict
        metrics: Dict[str, float] = {}
        if isinstance(out, dict):
            for k, v in out.items():
                try:
                    metrics[k] = float(v)
                except Exception:
                    pass
        return metrics


class ClientWMAla(ClientWMBase):
    """
    Skeleton for ALA-style client.
    In PFLlib, clientALA holds an ALA helper and exposes local_initialization(received_global_model)
    :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}.

    Here we keep the same hook so you can later drop in an ALA module that operates on WM params.
    """

    def __init__(self, cfg: Any, cid: int, tw: Any, ala_helper: Optional[Any] = None, **kwargs):
        super().__init__(cfg, cid, tw, **kwargs)
        self.ala = ala_helper  # optional external helper object

    def local_initialization(self, received_global_model: Dict[str, torch.Tensor]) -> None:
        if self.ala is None:
            return
        # Expect an interface similar to: ala.adaptive_local_aggregation(global_sd, local_model)
        # PFLlib calls: self.ALA.adaptive_local_aggregation(received_global_model, self.model)
        self.ala.adaptive_local_aggregation(received_global_model, self.tw.model)

    def train(self) -> Dict[str, float]:
        # After local init, training can be same as AVG by default
        local_steps = int(getattr(self.cfg.fl, "local_steps", 0))
        if local_steps <= 0:
            return {}

        out = self.tw.local_wm_train(local_steps, wm_config=self.cfg.wm)

        metrics: Dict[str, float] = {}
        if isinstance(out, dict):
            for k, v in out.items():
                try:
                    metrics[k] = float(v)
                except Exception:
                    pass
        return metrics


def make_client_class(fl_algo: str) -> Type[ClientWMBase]:
    """
    Small factory so main.py can select client class by name.
    Example:
      ClientCls = make_client_class(cfg.fl.algorithm)
      clients = [ClientCls(cfg, cid=i, tw=TwisterAdapter.from_config(...)) for i in ...]
    """
    name = (fl_algo or "").lower()
    if name in ("ala", "fedala", "clientala"):
        return ClientWMAla
    # default
    return ClientWMAvg
