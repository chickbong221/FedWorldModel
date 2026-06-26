#!/usr/bin/env python3
import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"


def _add_vendor_path() -> None:
    vendor = str(VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def _require(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise ModuleNotFoundError(
            f"Missing dependency: {name}. Install DreamerV3 requirements in a "
            "Python 3.11+ environment."
        )


def main() -> None:
    _add_vendor_path()

    print("python", sys.executable)
    print("version", sys.version.replace("\n", " "))
    print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

    for module in ["dreamerv3", "embodied", "jax", "elements", "portal", "ninjax"]:
        _require(module)
        print(f"{module}: ok")

    import jax

    print("jax", jax.__version__)
    print("jax_backend", jax.default_backend())
    print("jax_devices", jax.devices())
    if jax.default_backend() != "gpu":
        raise RuntimeError("JAX is not using GPU. Check CUDA/JAX installation.")


if __name__ == "__main__":
    main()

