# fedwm/client.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Type

import torch


class ClientWMBase:
    """
    FedWM version of PFLlib's flcore.clients.clientbase.Client :contentReference[oaicite:2]{index=2}
    but adapted to world-model training (TWISTER) instead of supervised classification.

    Design goals:
      1) main.py stays simple: set_global_wm -> (optional) local_initialization -> train -> package_update
      2) algorithm variants only override train() and/or local_initialization(), like clientAVG/clientALA
         :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}
    """

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
    def set_global_wm(self, wm_state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Receive global WM and load into local model.
        """
        self._global_wm_sd = wm_state_dict
        self.tw.set_wm_state_dict(wm_state_dict, strict=False)

    def set_policy(
        self,
        actor_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        critic_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Receive server actor/critic (optional).
        """
        self._actor_sd = actor_state_dict
        self._critic_sd = critic_state_dict
        if actor_state_dict is not None:
            self.tw.set_actor_state_dict(actor_state_dict, strict=False)
        if critic_state_dict is not None:
            self.tw.set_critic_state_dict(critic_state_dict, strict=False)

    # ---------- algorithm hooks ----------
    def local_initialization(self, received_global_model: Dict[str, torch.Tensor]) -> None:
        """
        Hook for algorithms like ALA that do a special local init *before* training,
        e.g. adaptive local aggregation. PFLlib clientALA uses this hook
        Default: do nothing.
        """
        return

    def train(self) -> Dict[str, float]:
        """
        Algorithm-specific local training.
        Must be overridden by subclasses (AVG, ALA, etc.).
        Returns metrics dict (optional).
        """
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
        """
        Return the update payload to the server.
        Full WM weights for now (easiest). Delta/compression can be added later.
        """
        if metrics is None:
            metrics = {}

        wm_sd = self.tw.get_wm_state_dict(to_cpu=True)

        return {
            "cid": self.id,
            "num_samples": self._num_samples_for_fedavg(),
            "wm_state_dict": wm_sd,
            "metrics": metrics,
        }

    def local_round(self) -> Dict[str, Any]:
        """
        The one function your current fedwm/main.py expects.

        Sequence:
          1) verify global received
          2) optional algorithm-specific init (ALA etc.)
          3) train locally
          4) package weights + stats
        """
        if self._global_wm_sd is None:
            raise RuntimeError(
                f"Client {self.id}: global WM not set. Call set_global_wm() before local_round()."
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
    Mirrors PFLlib's clientAVG where train() is the only override :contentReference[oaicite:6]{index=6}.
    """

    def train(self) -> Dict[str, float]:
        local_steps = int(getattr(self.cfg.fl, "local_steps", 0))
        if local_steps <= 0:
            return {}

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
        # :contentReference[oaicite:9]{index=9}
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
