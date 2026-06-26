# TWISTER Centralized Baseline

Run the centralized TWISTER baseline from the repository root:

```bash
conda activate fedwm
CUDA_VISIBLE_DEVICES=0 python wm_core/baselines/twister/train_centralized.py \
  --env_name atari100k-alien \
  --target_action_steps 400000 \
  --override_config '{"model_size":"SS"}'
```

