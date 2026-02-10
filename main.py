# fedwm/main.py
import argparse
import importlib
import json
import os
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch

# -------------------------
# utils
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _import_py_config(path: str):
    """
    path: 'configs/twister.py' or 'wm_core/configs/twister.py'
    returns imported module
    """
    mod = path.replace(".py", "").replace("/", ".")
    return importlib.import_module(mod)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


@dataclass
class Cfg:
    fl: SimpleNamespace
    wm: Any  # python module (TWISTER config module)


def build_cfg(fl_json_path: str, wm_config_py: str, override_json: Optional[str]) -> Cfg:
    fl_dict = _load_json(fl_json_path)
    if override_json:
        _deep_update(fl_dict, json.loads(override_json))

    wm_mod = _import_py_config(wm_config_py)
    return Cfg(fl=SimpleNamespace(**fl_dict), wm=wm_mod)


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    # --- configs (keep same vibe as your repos) ---
    parser.add_argument("--cfp", type=str, default="./hparams/FedAvg.json",
                        help="FL JSON config path (FedAvg-like).")
    parser.add_argument("--wm_config", type=str, default="wm_core/configs/twister.py",
                        help="TWISTER python config file.")
    parser.add_argument("--override", type=str, default=None,
                        help="JSON string to override FL fields, e.g. '{\"global_rounds\":10}'")
    parser.add_argument("--env_name", type=str, default=None,
                        help="Atari env/game name. If None, TWISTER may use its own default/env var.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    # Optional: keep TWISTER env var mechanism (TWISTER reads env_name/override_config in some setups)
    if args.env_name is not None:
        os.environ["env_name"] = args.env_name

    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")

    # ---- load configs ----
    cfg = build_cfg(args.cfp, args.wm_config, args.override)

    # ---- import your thin integration layer ----
    # These are in YOUR new repo (not in fl_core/wm_core)
    from fedwm.server import ServerWMAvg
    from fedwm.client import ClientWMAvg
    from fedwm.twister_adapter import TwisterAdapter

    # ---- build server (server owns global WM + actor/critic training) ----
    server_tw = TwisterAdapter.from_config(
        wm_config=cfg.wm,
        env_name=args.env_name,
        seed=args.seed,
        role="server",
    )
    server = ServerWMAvg(cfg=cfg, tw=server_tw)

    # ---- build clients (each client has its own WM + env + replay) ----
    clients = []
    for cid in range(int(cfg.fl.num_clients)):
        client_tw = TwisterAdapter.from_config(
            wm_config=cfg.wm,
            env_name=args.env_name,
            seed=args.seed + 1000 * cid,
            role=f"client{cid}",
        )
        clients.append(ClientWMAvg(cid=cid, cfg=cfg, tw=client_tw))

    # ---- federated training loop ----
    global_rounds = int(getattr(cfg.fl, "global_rounds", getattr(cfg.fl, "num_rounds", 100)))
    for r in range(global_rounds):
        print(f"\n===== Round {r+1}/{global_rounds} =====")

        server.reset_round_buffers()
        selected = server.select_clients(clients)
        wm_sd, actor_sd, critic_sd = server.get_payload()

        # broadcast + local client work
        for c in selected:
            c.set_client(wm_sd, actor_sd, critic_sd, reset_opt=False)
            c.tw.lease_to(device)
            try:
                upd = c.local_round()           # do WM env steps + WM updates on GPU
            finally:
                # Always park back to CPU and free VRAM even if local_round throws
                c.tw.sync_to_cpu(free_cuda=True)
            server.receive_update(upd)

        # aggregate world model
        server.aggregate_world_model()

        # server-side actor/critic training using global WM
        server.tw.lease_to(device)
        try:
            server.train_actor_critic()
        finally:
            server.tw.sync_to_cpu(free_cuda=True)

        server.log_round(r)

    print("\nAll done!")


if __name__ == "__main__":
    main()
