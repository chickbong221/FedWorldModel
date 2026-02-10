# fedwm/twister_adapter.py
# A thin wrapper around TWISTER so FL code can:
#   - construct per-client/server TWISTER instances
#   - get/set (world model / actor / critic) state_dict slices
#   - run a small number of local training steps using TWISTER's built-in fit() loop
#
# This adapter is based on TWISTER's entrypoint pattern (load config -> load model -> load dataset -> model.fit)
# shown in TWISTER main.py :contentReference[oaicite:0]{index=0} and the generic nnet Model.fit/train_step logic :contentReference[oaicite:1]{index=1}.

from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict, Optional, Tuple
from fedwm.utils import to_cpu_sd, to_device_sd, safe_int, safe_bool, set_seed
import nnet
import torch


def _ensure_wm_core_on_path() -> None:
    """
    Assumes repo layout:
      <repo_root>/
        fedwm/
        wm_core/
    and inserts <repo_root>/wm_core to sys.path so `import nnet` works.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    wm_core = os.path.join(repo_root, "wm_core")
    if wm_core not in sys.path:
        sys.path.insert(0, wm_core)


class TwisterAdapter:

    def __init__(self, model, replay_buffer, env_name, role):
        self.model = model
        self.replay_buffer = replay_buffer
        self.env_name = env_name
        self.role = role

        self._model_cpu_sd: Dict[str, torch.Tensor] = {}
        self._opt_cpu_sd: Optional[Dict[str, Any]] = None
        self._grad_scaler_cpu_sd: Optional[Dict[str, Any]] = None

        self._snapshot_to_cpu()
        self._park_model_to_cpu()

    # ----------------------------
    # internal
    # ----------------------------
    def _park_model_to_cpu(self):
        self.model.to("cpu")
        self.device = torch.device("cpu")

    def _snapshot_to_cpu(self, snapshot_opt: bool = True, snapshot_scaler: bool = True):
        # model weights always
        self._model_cpu_sd = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        if snapshot_opt:
            opt = getattr(self.model, "optimizer", None)
            if opt is not None:
                if isinstance(opt, dict):
                    opt_sd = {k: v.state_dict() for k, v in opt.items()}
                else:
                    opt_sd = opt.state_dict()
                self._opt_cpu_sd = to_cpu_sd(opt_sd)

        if snapshot_scaler:
            if hasattr(self.model, "grad_scaler"):
                try:
                    self._grad_scaler_cpu_sd = to_cpu_sd(self.model.grad_scaler.state_dict())
                except Exception:
                    self._grad_scaler_cpu_sd = None

    # ----------------------------
    # public API
    # ----------------------------
    def lease_to(self, device: torch.device):
        """
        CPU -> GPU:
          1) load CPU weights
          2) move model to device
          3) restore optimizer state (dict supported)
          4) restore grad scaler state (optional)
        """
        # 1) restore weights
        self.model.load_state_dict(self._model_cpu_sd, strict=True)

        # 2) move model
        self.model.to(device)
        self.device = device

        # 3) restore optimizer
        opt = getattr(self.model, "optimizer", None)
        if opt is not None and self._opt_cpu_sd is not None:
            opt_sd_dev = to_device_sd(self._opt_cpu_sd, device)
            if isinstance(opt, dict):
                for k, opt_k in opt.items():
                    if k in opt_sd_dev:
                        opt_k.load_state_dict(opt_sd_dev[k])
            else:
                opt.load_state_dict(opt_sd_dev)

        # 4) restore grad scaler
        if hasattr(self.model, "grad_scaler") and self._grad_scaler_cpu_sd is not None:
            try:
                self.model.grad_scaler.load_state_dict(self._grad_scaler_cpu_sd)
            except Exception:
                pass

    def sync_to_cpu(self, free_cuda: bool = True):
        """
        GPU -> CPU:
          snapshot model/opt/scaler to CPU, park on CPU, free VRAM.
        """
        self._snapshot_to_cpu()
        self._park_model_to_cpu()
        if free_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_cpu_model_state(self) -> Dict[str, torch.Tensor]:
        return self._model_cpu_sd

    def set_cpu_model_state(self, sd: Dict[str, torch.Tensor]):
        self._model_cpu_sd = {k: v.detach().cpu() for k, v in sd.items()}

    # ----------------------------
    # Construction
    # ----------------------------
    @staticmethod
    def from_config(
        wm_config: Any,
        env_name: Optional[str],
        seed: int,
        role: str,
        override_config: Optional[Dict[str, Any]] = None,
        buffer_root: Optional[str] = None,
    ) -> "TwisterAdapter":

        _ensure_wm_core_on_path()

        # env name
        if env_name is None:
            env_name = os.environ.get("env_name", None)
        if env_name is None:
            raise ValueError("env_name is required (pass --env_name or set env_name env var).")
        os.environ["env_name"] = env_name

        # role defaults
        role_lower = str(role).lower()
        role_override: Dict[str, Any] = {}
        if role_lower == "server":
            role_override["train_phase"] = "ac"
            role_override["do_env_step"] = True
        else:
            role_override["train_phase"] = "wm"
            role_override["do_env_step"] = True

        merged_override = {**role_override, **(override_config or {})}

        # env var override_config (highest priority)
        if os.environ.get("override_config", ""):
            try:
                oc = json.loads(os.environ["override_config"])
                if isinstance(oc, dict):
                    merged_override = {**merged_override, **oc}
            except Exception:
                pass

        # build on CPU (avoid VRAM blow-up)
        cpu = torch.device("cpu")
        model = nnet.models.TWISTER(env_name=env_name, override_config=merged_override)
        model.to(cpu)
        model.compile()  # creates optimizers etc. (stays on CPU)

        # replay buffer params
        batch_size = safe_int(getattr(wm_config, "batch_size", None), default=getattr(model.config, "batch_size", 16))
        buffer_capacity = safe_int(getattr(wm_config, "buffer_capacity", None), default=getattr(model.config, "buffer_capacity", 100_000))
        epoch_length = safe_int(getattr(wm_config, "epoch_length", None), default=getattr(model.config, "epoch_length", 1_000))
        sample_length = safe_int(getattr(wm_config, "L", None), default=getattr(model.config, "L", 64))
        shuffle = safe_bool(getattr(wm_config, "shuffle", None), default=True)

        if buffer_root is None:
            here = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(here)
            buffer_root = os.path.join(repo_root, "runs", "replay_buffers", env_name, role)
        os.makedirs(buffer_root, exist_ok=True)

        replay_buffer = nnet.datasets.ReplayBuffer(
            batch_size=batch_size,
            root=buffer_root,
            buffer_capacity=buffer_capacity,
            epoch_length=epoch_length,
            sample_length=sample_length,
            shuffle=shuffle,
            save_trajectories=safe_bool(getattr(wm_config, "save_trajectories", None), default=False),
            buffer_name="ReplayBuffer",
        )
        model.set_replay_buffer(replay_buffer)

        set_seed(seed)

        # adapter starts parked on CPU; device is the target for lease_to()
        return TwisterAdapter(
            model=model,
            replay_buffer=replay_buffer,
            env_name=env_name,
            role=role,
        )

    # ----------------------------
    # State dict slicing helpers
    # ----------------------------
    def _split_state_dict(
        self,
        sd: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if sd is None:
            sd = self._model_cpu_sd

        actor_prefixes = ("policy_network.",)
        critic_prefixes = ("value_network.", "v_target.")

        wm_sd: Dict[str, torch.Tensor] = {}
        actor_sd: Dict[str, torch.Tensor] = {}
        critic_sd: Dict[str, torch.Tensor] = {}

        for k, v in sd.items():
            if k.startswith(actor_prefixes):
                actor_sd[k] = v
            elif k.startswith(critic_prefixes):
                critic_sd[k] = v
            else:
                wm_sd[k] = v

        return wm_sd, actor_sd, critic_sd

    def get_payload(self, to_cpu: bool = True):
        # Split from CPU snapshot, NOT from live model
        wm_sd, actor_sd, critic_sd = self._split_state_dict(self._model_cpu_sd)

        if to_cpu:
            return (
                {k: v.detach().cpu() for k, v in wm_sd.items()},
                {k: v.detach().cpu() for k, v in actor_sd.items()},
                {k: v.detach().cpu() for k, v in critic_sd.items()},
            )
        else:
            return wm_sd, actor_sd, critic_sd
        
    def set_payload(
        self,
        wm_sd: Dict[str, torch.Tensor],
        actor_sd: Dict[str, torch.Tensor],
        critic_sd: Dict[str, torch.Tensor],
        strict: bool = False,
        reset_opt: bool = False,
    ):
        # Merge into authoritative CPU snapshot
        new_sd = dict(self._model_cpu_sd)  # start from existing snapshot
        new_sd.update({k: v.detach().cpu() for k, v in wm_sd.items()})
        new_sd.update({k: v.detach().cpu() for k, v in actor_sd.items()})
        new_sd.update({k: v.detach().cpu() for k, v in critic_sd.items()})

        self._model_cpu_sd = new_sd

        # Load into model (CPU) so future operations see the update
        self.model.load_state_dict(self._model_cpu_sd, strict=strict)

        # Optimizer handling
        if reset_opt:
            pass

    # ----------------------------
    # Training primitives
    # ----------------------------
    def local_wm_train(
        self,
        steps: int,
        wm_config: Any,
        *,
        precision: Optional[torch.dtype] = None,
        grad_init_scale: Optional[float] = None,
        accumulated_steps: int = 1,
    ) -> Dict[str, float]:

        steps = max(int(steps), 0)
        if steps == 0:
            return {}

        # Make 1 epoch == `steps` batches
        self.replay_buffer.epoch_length = steps

        # Pull defaults from config if not provided
        if precision is None:
            precision = getattr(wm_config, "precision", torch.float32)
        if grad_init_scale is None:
            grad_init_scale = getattr(wm_config, "grad_init_scale", 65536.0)

        self.model.fit(
            dataset_train=self.replay_buffer,
            epochs=1,
            dataset_eval=None,
            initial_epoch=0,
            callback_path=None,
            precision=precision,
            accumulated_steps=accumulated_steps,
            eval_period_step=None,
            eval_period_epoch=None,
            saving_period_epoch=10**9,
            log_figure_period_step=None,
            log_figure_period_epoch=10**9,
            step_log_period=10**9,
            grad_init_scale=grad_init_scale,
            detect_anomaly=False,
            recompute_metrics=False,
            wandb_logging=False,
            verbose_progress_bar=0,
            keep_last_k=None,
        )

        # Best-effort: TWISTER stores running infos internally, but not guaranteed stable API.
        # Return an empty dict for now; you can later parse self.model.infos / last logged losses.
        return {}

def _set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
