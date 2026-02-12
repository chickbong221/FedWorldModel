# fedwm/utils.py
import os
import sys
from typing import Any, Dict
import torch
import random
import numpy as np

def ensure_wm_core_on_path() -> str:
    """
    Repo layout:
      FedWorldModel/
        fedwm/
        wm_core/   <-- contains nnet/
    """
    here = os.path.dirname(os.path.abspath(__file__))     # .../FedWorldModel/fedwm
    repo_root = os.path.dirname(here)                     # .../FedWorldModel
    wm_core = os.path.join(repo_root, "wm_core")          # .../FedWorldModel/wm_core
    if wm_core not in sys.path:
        sys.path.insert(0, wm_core)
    return wm_core

# ============================
# State-dict helpers
# ============================

def to_cpu_sd(x: Any) -> Any:
    """Recursively move tensors in a state_dict to CPU."""
    if isinstance(x, dict):
        return {k: to_cpu_sd(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu_sd(v) for v in x)
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x


def to_device_sd(x: Any, device: torch.device) -> Any:
    """Recursively move tensors in a state_dict to a device."""
    if isinstance(x, dict):
        return {k: to_device_sd(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device_sd(v, device) for v in x)
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    return x


# ============================
# Config helpers
# ============================

def safe_int(x, default: int):
    try:
        return int(x)
    except Exception:
        return default


def safe_bool(x, default: bool):
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = str(x).lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


# ============================
# Seed helper
# ============================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)