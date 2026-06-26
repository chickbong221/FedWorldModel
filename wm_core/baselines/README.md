# World Model Baselines

This directory keeps baseline entrypoints separate from the main FedWorldModel
and federated-learning code.

- `twister/`: centralized TWISTER baseline using the existing `wm_core` model.
- `dreamerv3/`: vendored Danijar DreamerV3 baseline and local runner.

Install baseline dependencies from the shared requirements file:

```bash
pip install -r wm_core/baselines/requirements.txt
```
