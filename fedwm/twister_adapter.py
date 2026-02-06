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
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_bool(x: Any, default: bool) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.lower() in ("1", "true", "yes", "y", "t")
    return default


@dataclass
class TwisterAdapter:
    """
    Wraps a TWISTER model instance + its ReplayBuffer.

    Notes:
    - TWISTER defines set_replay_buffer() which also resets env + seeds episode history,
      and appends the reset step into the buffer.
    - TWISTER's training is normally done via model.fit(dataset_train=..., epochs=..., ...)
      as in their main.py.
    """

    model: torch.nn.Module
    replay_buffer: Any
    device: torch.device
    env_name: str
    role: str

    # ----------------------------
    # Construction
    # ----------------------------
    @staticmethod
    def from_config(
        wm_config: Any,
        device: torch.device,
        env_name: Optional[str],
        seed: int,
        role: str,
        override_config: Optional[Dict[str, Any]] = None,
        buffer_root: Optional[str] = None,
    ) -> "TwisterAdapter":
        """
        Create a TWISTER instance and its ReplayBuffer.

        - wm_config is the imported TWISTER python config module (e.g. configs/twister.py)
          like TWISTER main.py does via importlib :contentReference[oaicite:4]{index=4}.
        - override_config is passed to TWISTER constructor (TWISTER repo supports this pattern).
        """

        _ensure_wm_core_on_path()

        # Try to import TWISTER in the most compatible way for your merged repo.
        # In the TWISTER repo, the usual entry is `import nnet` then `nnet.models.TWISTER(...)`.
        import nnet  # type: ignore

        # Env name: TWISTER often reads from env var in some setups; we keep both.
        if env_name is None:
            env_name = os.environ.get("env_name", None)
        if env_name is None:
            raise ValueError("env_name is required (pass --env_name or set env_name env var).")
        os.environ["env_name"] = env_name

        # Override config: allow caller to pass a dict; also allow env var override_config
        if override_config is None:
            override_config = {}
        # Some of your earlier code uses override_config env var (stringified JSON); keep compatible.
        if "override_config" in os.environ and os.environ["override_config"]:
            # best-effort parse; ignore if malformed
            import json

            try:
                oc = json.loads(os.environ["override_config"])
                if isinstance(oc, dict):
                    override_config = {**override_config, **oc}
            except Exception:
                pass

        # Build TWISTER model
        # NOTE: TWISTER's main uses functions.load_model(args), but the public pattern is:
        #   model = nnet.models.TWISTER(env_name=..., override_config=...)
        #   model.compile()
        model = nnet.models.TWISTER(env_name=env_name, override_config=override_config)
        model.to(device)
        model.compile()

        # ---- replay buffer params (taken from config module if present; otherwise safe defaults) ----
        # ReplayBuffer signature shows required args :contentReference[oaicite:6]{index=6}
        from replay_buffer import ReplayBuffer  # type: ignore

        batch_size = _safe_int(getattr(wm_config, "batch_size", None), default=16)
        buffer_capacity = _safe_int(getattr(wm_config, "buffer_capacity", None), default=100_000)
        epoch_length = _safe_int(getattr(wm_config, "epoch_length", None), default=1_000)
        sample_length = _safe_int(getattr(wm_config, "L", None), default=64)
        shuffle = _safe_bool(getattr(wm_config, "shuffle", None), default=True)

        # buffer_root: separate per-role buffers to avoid collision on disk
        if buffer_root is None:
            # keep buffers inside repo_root/runs/replay_buffers/<env_name>/<role>/
            here = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(here)
            buffer_root = os.path.join(repo_root, "runs", "replay_buffers", env_name, role)
        os.makedirs(buffer_root, exist_ok=True)

        replay_buffer = ReplayBuffer(
            batch_size=batch_size,
            root=buffer_root,
            buffer_capacity=buffer_capacity,
            epoch_length=epoch_length,
            sample_length=sample_length,
            shuffle=shuffle,
            save_trajectories=_safe_bool(getattr(wm_config, "save_trajectories", None), default=False),
            buffer_name="ReplayBuffer",
        )

        # Attach buffer to TWISTER (resets env and appends reset step) :contentReference[oaicite:7]{index=7}
        model.set_replay_buffer(replay_buffer)

        # Seed (TWISTER sets seeds in main.py; we do minimal local seeding)
        _set_seed(seed)

        return TwisterAdapter(
            model=model,
            replay_buffer=replay_buffer,
            device=device,
            env_name=env_name,
            role=role,
        )

    # ----------------------------
    # State dict slicing helpers
    # ----------------------------
    def _split_state_dict(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Split full model state into:
          - world model (encoder/rssm/decoder/reward/continue + anything not actor/critic)
          - actor (policy_network.*)
          - critic (value_network.* and v_target.* if present)
        Module names are visible in TWISTER (e.g. encoder_network, rssm, policy_network, value_network) :contentReference[oaicite:8]{index=8}.
        """
        sd = self.model.state_dict()
        actor_prefixes = ("policy_network.",)
        critic_prefixes = ("value_network.", "v_target.")
        actor_sd: Dict[str, torch.Tensor] = {}
        critic_sd: Dict[str, torch.Tensor] = {}
        wm_sd: Dict[str, torch.Tensor] = {}

        for k, v in sd.items():
            if k.startswith(actor_prefixes):
                actor_sd[k] = v
            elif k.startswith(critic_prefixes):
                critic_sd[k] = v
            else:
                wm_sd[k] = v

        return wm_sd, actor_sd, critic_sd

    def get_wm_state_dict(self, to_cpu: bool = True) -> Dict[str, torch.Tensor]:
        wm_sd, _, _ = self._split_state_dict()
        return {k: (v.detach().cpu() if to_cpu else v.detach()) for k, v in wm_sd.items()}

    def set_wm_state_dict(self, wm_sd: Dict[str, torch.Tensor], strict: bool = False) -> None:
        # load partial dict
        self.model.load_state_dict(wm_sd, strict=strict)

    def get_actor_state_dict(self, to_cpu: bool = True) -> Dict[str, torch.Tensor]:
        _, actor_sd, _ = self._split_state_dict()
        return {k: (v.detach().cpu() if to_cpu else v.detach()) for k, v in actor_sd.items()}

    def set_actor_state_dict(self, actor_sd: Dict[str, torch.Tensor], strict: bool = False) -> None:
        self.model.load_state_dict(actor_sd, strict=strict)

    def get_critic_state_dict(self, to_cpu: bool = True) -> Dict[str, torch.Tensor]:
        _, _, critic_sd = self._split_state_dict()
        return {k: (v.detach().cpu() if to_cpu else v.detach()) for k, v in critic_sd.items()}

    def set_critic_state_dict(self, critic_sd: Dict[str, torch.Tensor], strict: bool = False) -> None:
        self.model.load_state_dict(critic_sd, strict=strict)

    # Convenience used by your fedwm.server/client
    def get_payload(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        return (self.get_wm_state_dict(to_cpu=True),
                self.get_actor_state_dict(to_cpu=True),
                self.get_critic_state_dict(to_cpu=True))

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
        """
        Run a small amount of TWISTER training using its built-in Model.fit loop.

        Why this works:
        - ReplayBuffer.__len__ = epoch_length * batch_size
        - Model.fit iterates over dataset_train batches and calls train_step repeatedly
        So if we set replay_buffer.epoch_length = steps, one "epoch" corresponds to ~steps gradient steps.

        NOTE: This will update *all* learnable params (wm + actor + critic) unless you freeze modules.
        For an MVP, it's OK; later you can add freeze_world_model()/freeze_actor_critic() controls.
        """
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

        # Turn off heavy stuff (callbacks, wandb, eval) for local steps.
        # TWISTER main calls model.fit(...) with many knobs.
        # We pass the minimal safe subset supported by the base Model.fit implementation.
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

    # ----------------------------
    # Optional: freezing helpers (handy for server_ac_updates later)
    # ----------------------------
    def set_requires_grad_world_model(self, enabled: bool) -> None:
        """
        Toggle gradients for common WM submodules.
        Names match TWISTER fields (encoder_network, rssm, decoder_network, reward_network, continue_network) :contentReference[oaicite:13]{index=13}.
        """
        for name in ("encoder_network", "rssm", "decoder_network", "reward_network", "continue_network"):
            m = getattr(self.model, name, None)
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad = enabled

    def set_requires_grad_actor_critic(self, enabled: bool) -> None:
        for name in ("policy_network", "value_network", "v_target"):
            m = getattr(self.model, name, None)
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad = enabled

    def actor_critic_train_steps(self, steps: int, wm_config: Any) -> dict:
        self.set_requires_grad_world_model(False)
        self.set_requires_grad_actor_critic(True)
        out = self.local_wm_train(steps, wm_config=wm_config, freeze_actor_critic=False)
        self.set_requires_grad_world_model(True)
        self.set_requires_grad_actor_critic(True)
        return out


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
