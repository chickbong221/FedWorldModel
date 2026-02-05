# Copyright 2025, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
from torch import nn
import torchvision
import random

# NeuralNets
from nnet import models
from nnet import optimizers
from nnet import envs
from nnet.modules import twister as twister_networks
from nnet.structs import AttrDict

# Other
import copy
import itertools
import os
import glob
import torch.nn.functional as F

class TWISTER(models.Model):

    def __init__(self, env_name, override_config={}, name="Transformer-based World model wIth contraSTivE Representations (TWISTER)"):
        super(TWISTER, self).__init__(name=name)

        # Model Sizes
        model_sizes = {
            "S": AttrDict({
                "dim_cnn": 32,
                "hidden_size": 512,
                "num_layers": 2,

                "stoch_size": 32,
                "discrete": 32,

                "num_blocks_trans": 4,
                "ff_ratio_trans": 2,
                "num_heads_trans": 8,
                "drop_rate_trans": 0.1
            }),
        }

        # Env Type
        env_name = env_name.split("-")
        self.env_type = env_name[0]
        assert self.env_type in ["dmc", "atari100k"]

        # Config
        self.config = AttrDict()
        self.config.env_name = env_name
        self.config.env_type = env_name[0]

        # Env
        if self.env_type == "dmc":
            self.config.env_class = envs.dm_control.dm_control_dict[env_name[1]]
            self.config.env_params = {"task": env_name[2], "history_frames": 1, "img_size": (64, 64), "action_repeat": 2}
            self.config.model_size = "S"
            self.config.time_limit = 1000
            self.config.time_limit_eval = 1000
        elif self.env_type == "atari100k":
            self.config.env_class = envs.atari.AtariEnv
            self.config.env_params = {"game": env_name[1], "history_frames": 1, "img_size": (64, 64), "action_repeat": 4, "grayscale_obs": False, "noop_max": 30, "repeat_action_probability": 0.0, "full_action_space": False}
            self.config.model_size = "S"
            self.config.time_limit = 108000
            self.config.time_limit_eval = 108000
        self.config.eval_env_params = {}
        self.config.train_env_params = {}

        # Training
        self.config.batch_size = 16
        self.config.L = 64
        self.config.H = 15
        self.config.num_envs = {"dmc": 4, "atari100k": 1}[self.env_type]
        self.config.epochs = {"dmc": 50, "atari100k": 50}[self.env_type]
        self.config.epoch_length = {"dmc": 5000, "atari100k": 2000}[self.env_type]
        self.config.env_step_period = {"dmc": 512, "atari100k": 1024}[self.env_type] # call env_step (batch_size * L) / (env_step_period * num_envs) times

        # Eval
        self.config.eval_episodes = {"dmc": 10, "atari100k": 100}[self.env_type]

        # Optimizer
        self.config.opt_weight_decay = 0.0
        self.config.model_lr = 1e-4
        self.config.critic_lr = 3e-5
        self.config.actor_lr = 3e-5
        self.config.model_eps = 1e-8
        self.config.critic_eps = 1e-5
        self.config.actor_eps = 1e-5
        self.config.model_grad_max_norm = 1000
        self.config.critic_grad_max_norm = 100
        self.config.actor_grad_max_norm = 100
        self.config.grad_init_scale = 32.0
        self.config.precision = {"dmc": torch.float16, "atari100k": torch.float32}[self.env_type]

        # Replay Buffer
        self.config.buffer_capacity = int(1e6)
        self.config.pre_fill_steps = 100 # pre_fill_steps in number of buffer samples
        self.config.load_replay_buffer_state_dict = True # Load ReplayBuffer saved state dict from checkpoint

        # Return Norm
        self.config.return_norm_decay = 0.99
        self.config.return_norm_limit = 1.0
        self.config.return_norm_perc_low = 0.05
        self.config.return_norm_perc_high = 0.95

        # World Model Params
        model_params = model_sizes[self.config.model_size]
        self.config.norm = {"class": "LayerNorm", "params": {"eps": 1e-3, "convert_float32": True}}
        self.config.free_nats = 1.0
        self.config.image_channels = 3
        self.config.dim_cnn = model_params.dim_cnn
        self.config.repr_layers = model_params.num_layers
        self.config.repr_hidden_size = model_params.hidden_size
        self.config.model_discrete = model_params.discrete
        self.config.model_stoch_size = model_params.stoch_size
        self.config.model_hidden_size = model_params.hidden_size
        self.config.action_hidden_size = model_params.hidden_size
        self.config.value_hidden_size = model_params.hidden_size
        self.config.reward_hidden_size = model_params.hidden_size
        self.config.discount_hidden_size = model_params.hidden_size
        self.config.action_layers = model_params.num_layers
        self.config.value_layers = model_params.num_layers
        self.config.reward_layers = model_params.num_layers
        self.config.discount_layers = model_params.num_layers
        self.config.learn_initial = True

        # Actor Params
        self.config.actor_grad = "reinforce"
        self.config.policy_discrete = {"dmc": False, "atari100k": True}[self.env_type]
        self.config.eta_entropy = 0.0003
        self.config.sampling_tmp = 1.0

        # Critic Params
        self.config.lambda_td = 0.95
        self.config.gamma = 0.997
        self.config.target_value_reg = True
        self.config.critic_ema_decay = 0.02
        self.config.critic_slow_reg_scale = 1.0

        # Loss Scales
        self.config.loss_reward_scale = 1.0
        self.config.loss_discount_scale = 1.0
        self.config.loss_decoder_scale = 1.0
        self.config.loss_kl_prior_scale = 0.5
        self.config.loss_kl_post_scale = 0.1
        self.config.loss_contrastive_scale = 0.3

        # TSSM
        self.config.att_context_left = 8 # C must be <= L
        self.config.num_blocks_trans = model_params.num_blocks_trans
        self.config.ff_ratio_trans = model_params.ff_ratio_trans
        self.config.num_heads_trans = model_params.num_heads_trans
        self.config.drop_rate_trans = model_params.drop_rate_trans
        self.config.encoder_cnn_norm = {"class": "LayerNorm", "params": {"eps": 1e-3, "convert_float32": True}}
        self.config.module_pre_norm = False
        self.config.detach_decoder = False

        # Contrastive
        self.config.contrastive_augments = torchvision.transforms.RandomResizedCrop(size=(64, 64), antialias=True, scale=(0.25, 1))
        self.config.contrastive_hidden_size = self.config.model_hidden_size
        self.config.contrastive_out_size = self.config.contrastive_hidden_size
        self.config.contrastive_steps = 1
        self.config.contrastive_exp_lambda = 0.75
        self.config.contrastive_layers = 2

        # Adversarial
        self.config.window_size = [4, 16, 32]
        self.config.num_seq_to_discriminate = [8, 8, 4]
        self.config.adversarial_hidden_dim = [192, 256, 384]
        self.config.adversarial_proj_dim = [128, 192, 256]
        self.config.adversarial_num_heads = [2, 4, 6]
        self.config.adversarial_num_layers = [1, 1, 2]
        self.config.adversarial_max_perms = 1000

        # Sample Pre Fill Steps
        self.config.random_pre_fill_steps = True

        # Log Figure
        self.config.log_figure_batch = 16
        self.config.log_figure_context_frames = 5

        # Override Config
        for key, value in override_config.items():
            assert key in self.config, "{} not in config".format(key)

            if key=="precision":
                self.config[key] = {"float16": torch.float16, "float32": torch.float32}[value]
            else:
                self.config[key] = value

        # Config asserts
        assert self.config.att_context_left <= self.config.L

        # Create Training Envs
        self.env = envs.wrappers.BatchEnv([
            envs.wrappers.ResetOnException(
                envs.wrappers.TimeLimit(
                    self.config.env_class(**dict(self.config.env_params, **self.config.train_env_params)), 
                    time_limit=self.config.time_limit_eval
                )
            )
        for env_i in range(self.config.num_envs)])
                
        # Create Evaluation Env
        if self.config.eval_episodes > 0:
            self.env_eval = envs.wrappers.ResetOnException(
                envs.wrappers.TimeLimit(
                    self.config.env_class(**dict(self.config.env_params, **self.config.eval_env_params)),
                    time_limit=self.config.time_limit_eval
                )
            )
        else:
            self.env_eval = None

        # Networks
        feat_size = self.config.model_stoch_size * self.config.model_discrete + self.config.model_hidden_size if self.config.model_discrete else self.config.model_stoch_size + self.config.model_hidden_size
        self.encoder_network = twister_networks.EncoderNetwork(
            dim_input_cnn=self.config.image_channels, 
            dim_cnn=self.config.dim_cnn,
            cnn_norm=self.config.encoder_cnn_norm,
            stoch_size=self.config.model_stoch_size,
            discrete=self.config.model_discrete,
        )
        self.decoder_network = twister_networks.DecoderNetwork(
            dim_output_cnn=self.config.image_channels, 
            feat_size=self.config.model_stoch_size * self.config.model_discrete, 
            dim_cnn=self.config.dim_cnn, 
            cnn_norm=self.config.norm,
        )
        self.rssm = twister_networks.TSSM(
            num_actions=self.env.num_actions, 
            stoch_size=self.config.model_stoch_size, 
            discrete=self.config.model_discrete, 
            learn_initial=self.config.learn_initial,
            norm=self.config.norm,
            hidden_size=self.config.model_hidden_size,
            num_blocks=self.config.num_blocks_trans,
            ff_ratio=self.config.ff_ratio_trans,
            num_heads=self.config.num_heads_trans,
            drop_rate=self.config.drop_rate_trans,
            att_context_left=self.config.att_context_left,
            module_pre_norm=self.config.module_pre_norm
        )
        self.policy_network = twister_networks.PolicyNetwork(
            num_actions=self.env.num_actions, 
            hidden_size=self.config.action_hidden_size, 
            feat_size=feat_size, 
            num_mlp_layers=self.config.action_layers, 
            discrete=self.config.policy_discrete,
            norm=self.config.norm,
            sampling_tmp=self.config.sampling_tmp
        )
        self.value_network = twister_networks.ValueNetwork(
            hidden_size=self.config.value_hidden_size, 
            feat_size=feat_size, 
            num_mlp_layers=self.config.value_layers,
            norm=self.config.norm
        )
        self.reward_network = twister_networks.RewardNetwork(
            hidden_size=self.config.reward_hidden_size, 
            feat_size=feat_size, 
            num_mlp_layers=self.config.reward_layers,
            norm=self.config.norm
        )
        self.continue_network = twister_networks.ContinueNetwork(
            hidden_size=self.config.discount_hidden_size, 
            feat_size=feat_size, 
            num_mlp_layers=self.config.discount_layers,
            norm=self.config.norm
        )
        # self.contrastive_network = nn.ModuleList([twister_networks.ContrastiveNetwork(
        #     feat_size=feat_size + t * self.env.num_actions,
        #     embed_size=self.config.model_stoch_size * self.config.model_discrete,
        #     hidden_size=self.config.contrastive_hidden_size,
        #     out_size=self.config.contrastive_out_size,
        #     num_layers=self.config.contrastive_layers
        # ) for t in range(self.config.contrastive_steps)])

        # self.discriminator_network = twister_networks.TemporalDiscriminator(
        #     num_layers=4,
        #     num_heads=4,
        #     dropout=0.1
        # )

        self.discriminator_network = nn.ModuleList([
            twister_networks.TemporalDiscriminator(
                proj_dim=self.config.adversarial_proj_dim[i],
                hidden_dim=self.config.adversarial_hidden_dim[i],
                num_layers=self.config.adversarial_num_layers[i],
                num_heads=self.config.adversarial_num_heads[i],
                dropout=0.05,
            )
            for i in range(len(self.config.window_size))
        ])

        # self.temporal_order_discriminator = nn.ModuleList([twister_networks.TemporalOrderDiscriminator(
        #     feat_dim=256,
        #     window_size=w,
        #     hidden_dim=256,
        #     num_heads=4,
        #     num_layers=2,
        #     max_perms=1000
        # ) for w in self.config.window_size])
        
        def count_parameters(model_list):
            total_params = sum(p.numel() for model in model_list for p in model.parameters())
            print(f"Total parameters in ModuleList: {total_params:,}")

        # count_parameters(self.temporal_order_discriminator)

        # Slow Moving Networks
        self.add_frozen("v_target", copy.deepcopy(self.value_network))

        # Percentiles
        self.register_buffer("perc_low", torch.tensor(0.0))
        self.register_buffer("perc_high", torch.tensor(0.0))
        
        # Training Infos
        self.register_buffer("episodes", torch.tensor(0))
        self.register_buffer("ep_rewards", torch.zeros(self.config.num_envs), persistent=False)
        self.register_buffer("action_step", torch.tensor(0))

        # World Model
        self.world_model = self.WorldModel(outer=self)

        # Actor Model
        self.actor_model = self.ActorModel(outer=self)

        # Critic Model
        self.critic_model = self.CriticModel(outer=self)

    def summary(self, show_dict=False, show_modules=False):

        # Model Name
        print("Model name: {}".format(self.name))

        # Number Params
        print("World Model Parameters: {:,}".format(self.num_params(self.world_model)))
        print("Actor Parameters: {:,}".format(self.num_params(self.actor_model)))
        print("Critic Parameters: {:,}".format(self.num_params(self.critic_model)))

        # Options
        if show_dict:
            self.show_dict()
        if show_modules:
            self.show_modules()

    def preprocess_inputs(self, state, time_stacked):

        def norm_image(image):

            assert image.dtype == torch.uint8

            return image.type(torch.float32) / 255 - 0.5

        # List of Inputs
        if isinstance(state, list):
            state = [norm_image(s) if s.dim()==(5 if time_stacked else 4) else s for s in state]

        # State (could be image or lowd)
        else:
            state = norm_image(state) if state.dim()==(5 if time_stacked else 4) else state

        return state

    def save(self, path, save_optimizer=True, keep_last_k=None):
        
        # Save Model Checkpoint
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": None if not save_optimizer else {key: value.state_dict() for key, value in self.optimizer.items()} if isinstance(self.optimizer, dict) else self.optimizer.state_dict(),
            "model_step": self.model_step,
            "grad_scaler_state_dict": self.grad_scaler.state_dict() if hasattr(self, "grad_scaler") else None,
            "replay_buffer_state_dict": self.replay_buffer.state_dict()
        }, path)
        
        # Save Buffer
        self.replay_buffer.save()

        # Print Model state
        print("Model saved at step {}: {}".format(self.model_step, path))

        # Keep last k checkpoints
        if keep_last_k != None:

            # List checkpoints
            save_dir = os.path.dirname(path)
            checkpoints_list = glob.glob(os.path.join(save_dir, "*.ckpt"))
            checkpoints_list = sorted(checkpoints_list, key=lambda s: int(os.path.splitext(s)[0].split("/")[-1].split("_")[-1]))

            # Remove older_checkpoint
            while len(checkpoints_list) > keep_last_k:

                # Pop older_checkpoint
                older_checkpoint = checkpoints_list.pop(0)

                # Remove older_checkpoint
                os.remove(older_checkpoint)

                # Print
                print("Removed old checkpoint {}".format(older_checkpoint))

    def load(self, path, load_optimizer=True, verbose=True, strict=True):

        # Print Load state
        if verbose:
            print("Load Model from {}".format(path))

        # Load Model Checkpoint
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load Model State Dict
        self.load_state_dict({key:value for key, value in checkpoint["model_state_dict"].items()}, strict=strict)

        # Load Optimizer State Dict
        if load_optimizer and checkpoint["optimizer_state_dict"] is not None:

            if isinstance(self.optimizer, dict):
                for key, value in self.optimizer.items():
                    value.load_state_dict(checkpoint["optimizer_state_dict"][key])
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Model Step, already loaded from optm
            self.model_step.fill_(checkpoint["model_step"])

        # Load replay Buffer State Dict
        if self.config.load_replay_buffer_state_dict:
            self.replay_buffer.load_state_dict(checkpoint["replay_buffer_state_dict"])
        elif verbose:
            print("load_replay_buffer_state_dict set to False: replay buffer state dict not loaded")

        # Load Grad Scaler
        if "grad_scaler_state_dict" in checkpoint:
            self.grad_scaler_state_dict = checkpoint["grad_scaler_state_dict"]

        # Print Model state
        if verbose:
            print("Model loaded at step {}".format(self.model_step))

    def set_replay_buffer(self, replay_buffer):

        # Replay Buffer
        self.replay_buffer = replay_buffer

        # Set History
        obs_reset = self.env.reset()
        self.episode_history = AttrDict(
            ep_step=torch.zeros(self.config.num_envs), # (N,)
            hidden=(self.rssm.initial(batch_size=self.config.num_envs, seq_length=2, dtype=torch.float32, detach_learned=True), torch.zeros(self.config.num_envs, self.env.num_actions, dtype=torch.float32)), 
            state=obs_reset.state,
            episodes=[AttrDict(
                states=[obs_reset.state[env_i]],
                actions=[torch.zeros(self.env.num_actions, dtype=torch.float32)],
                rewards=[obs_reset.reward[env_i]],
                dones=[obs_reset.done[env_i]],
                is_firsts=[obs_reset.is_first[env_i]],
                model_steps=[self.model_step.clone()]
            ) for env_i in range(self.config.num_envs)]
        )

        # Update Buffer with reset step
        for env_i in range(self.config.num_envs):
            sample = []
            sample.append(obs_reset.state[env_i])
            sample.append(torch.zeros(self.env.num_actions, dtype=torch.float32))
            sample.append(obs_reset.reward[env_i])
            sample.append(obs_reset.done[env_i])
            sample.append(obs_reset.is_first[env_i])
            sample.append(self.model_step.clone())
            buffer_infos = self.replay_buffer.append_step(sample, env_i)

        # Add Buffer Infos
        for key, value in buffer_infos.items():
            self.add_info(key, value)

    def on_train_begin(self):

        # Pre Fill Buffer
        if self.config.pre_fill_steps > 0 and self.replay_buffer.num_steps < self.config.pre_fill_steps:
            print("Prefill dataset with {} steps, policy={}".format(self.config.pre_fill_steps, "random" if self.config.random_pre_fill_steps else "sample"))
            while self.replay_buffer.num_steps < self.config.pre_fill_steps:
                self.env_step()

    def compile(self):
        
        # Compile World Model
        model_params = itertools.chain(self.encoder_network.parameters(), self.rssm.parameters(), self.reward_network.parameters(), self.decoder_network.parameters(), self.continue_network.parameters())
        self.world_model.compile(
            optimizer=optimizers.Adam(params=[
                {"params": model_params, "lr": self.config.model_lr, "grad_max_norm": self.config.model_grad_max_norm, "eps": self.config.model_eps}, 
            ], weight_decay=self.config.opt_weight_decay), 
            losses={},
            loss_weights={},
            metrics=None,
            decoders=None
        )

        # Compile Actor Model
        self.actor_model.compile(
            optimizer=optimizers.Adam(params=[
                {"params": self.policy_network.parameters(), "lr": self.config.actor_lr, "grad_max_norm": self.config.actor_grad_max_norm, "eps": self.config.actor_eps},
            ], weight_decay=self.config.opt_weight_decay), 
            losses={},
            loss_weights={},
            metrics=None,
            decoders=None
        )

        # Compile Critic Model
        self.critic_model.compile(
            optimizer=optimizers.Adam(params=[
                {"params": self.value_network.parameters(), "lr": self.config.critic_lr, "grad_max_norm": self.config.critic_grad_max_norm, "eps": self.config.critic_eps}, 
            ], weight_decay=self.config.opt_weight_decay), 
            losses={},
            loss_weights={},
            metrics=None,
            decoders=None
        )

        # self.discriminator_network.opt_disc = torch.optim.Adam(self.discriminator_network.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.discriminator_optimizers = [
            torch.optim.Adam(
                discriminator.parameters(),
                lr=2e-4,
                betas=(0.5, 0.999),
                eps=self.config.model_eps
            )
            for discriminator in self.discriminator_network
        ]

        # self.tod_optimizers = [
        #     torch.optim.Adam(
        #         tod.parameters(),
        #         lr=2e-4,
        #         betas=(0.5, 0.999),
        #         eps=self.config.model_eps
        #     )
        #     for tod in self.temporal_order_discriminator
        # ]

        # Model Step
        self.model_step = self.world_model.optimizer.param_groups[0]["lr_scheduler"].model_step

        # Optimizer
        self.optimizer = {"world_model": self.world_model.optimizer, "actor_model": self.actor_model.optimizer, "critic_model": self.critic_model.optimizer}

        # Set Compiled to True
        self.compiled = True

    def env_step(self):

        # Eval Mode
        training = self.training
        self.encoder_network.eval()
        self.rssm.eval()
        self.policy_network.eval()

        ###############################################################################
        # Forward / Env Step
        ###############################################################################

        # Recover State / hidden
        state = self.episode_history.state
        hidden = self.episode_history.hidden

        # Unpack hidden
        prev_latent, action = hidden

        # Transfer to device
        state = self.transfer_to_device(state)
        prev_latent = self.transfer_to_device(prev_latent)
        action = self.transfer_to_device(action)

        # Forward Policy Network
        with torch.no_grad():

            # Repr State (B, ...)
            latent = self.encoder_network(self.preprocess_inputs(state, time_stacked=False))

            # Unsqueeze Time dim (B, 1, ...)
            latent = {key: value.unsqueeze(dim=1) for key, value in latent.items()}

            # Generate is_firsts_hidden for forward
            if prev_latent["hidden"] != None:
                is_firsts_hidden = torch.zeros(self.config.num_envs, 
                                            self.rssm.get_hidden_len(prev_latent["hidden"]), 
                                            dtype=torch.float32, device=action.device)
                for env_i in range(self.config.num_envs):
                    env_i_length = len(self.episode_history.episodes[env_i].is_firsts) - 1
                    if 0 < env_i_length <= is_firsts_hidden.shape[1]:
                        is_firsts_hidden[env_i, -env_i_length] = 1.0
            else:
                is_firsts_hidden = None

            # RSSM (B, 1, ...)
            # Note: prev_latent already contains 2 timesteps from initialization
            latent, _ = self.rssm(
                states=latent, 
                prev_states=prev_latent, 
                prev_actions=action.unsqueeze(dim=1), 
                is_firsts=torch.tensor([1.0 if len(self.episode_history.episodes[env_i].is_firsts) == 1 
                                    else 0.0 for env_i in range(self.config.num_envs)], 
                                    dtype=torch.float32, device=action.device).unsqueeze(dim=1),
                is_firsts_hidden=is_firsts_hidden
            )

            # Get feat (B, Dfeat) - use the most recent timestep
            feat = self.rssm.get_feat(latent).squeeze(dim=1)

            # Policy Sample
            action = self.policy_network(feat).sample().cpu()

        # Update Hidden - maintain 2-timestep window for motion
        # Shift window: drop oldest, keep recent, add new
        latent_for_hidden = {}
        for key in latent.keys():
            if key == "hidden":
                latent_for_hidden[key] = self.rssm.slice_hidden(latent[key])
            else:
                # Create 2-timestep window: [prev_latent's recent, new latent]
                # prev_latent has shape (B, 2, ...), we take index 1 (most recent)
                # new latent has shape (B, 1, ...), we concatenate them
                latent_for_hidden[key] = torch.cat([
                    prev_latent[key][:, -1:],  # Most recent from previous
                    latent[key]  # New latent
                ], dim=1)  # Result: (B, 2, ...)
        
        hidden = (latent_for_hidden, action)

        # Clip Action
        if not self.config.policy_discrete:
            action = action.type(torch.float32).clip(self.env.clip_low, self.env.clip_high)

        # Env Step
        if (self.replay_buffer.num_steps < self.config.pre_fill_steps) and self.config.random_pre_fill_steps:
            action = self.env.sample()
        obs = self.env.step(action.argmax(dim=-1) if self.config.policy_discrete else action)

        ###############################################################################
        # Update Infos / Buffer
        ###############################################################################

        # Update training_infos
        self.action_step += self.env.action_repeat * self.config.num_envs
        self.ep_rewards += obs.reward.to(self.ep_rewards.device)

        # Update History State
        self.episode_history.state = obs.state
        self.episode_history.hidden = hidden
        self.episode_history.ep_step += self.env.action_repeat
        
        # Update History Episodes
        for env_i in range(self.config.num_envs):
            if not obs.error[env_i]:
                self.episode_history.episodes[env_i].states.append(obs.state[env_i])
                self.episode_history.episodes[env_i].actions.append(action[env_i])
                self.episode_history.episodes[env_i].rewards.append(obs.reward[env_i])
                self.episode_history.episodes[env_i].dones.append(obs.done[env_i])
                self.episode_history.episodes[env_i].is_firsts.append(obs.is_first[env_i])
                self.episode_history.episodes[env_i].model_steps.append(self.model_step.clone())

        # Update Traj Buffer
        for env_i in range(self.config.num_envs):
            if not obs.error[env_i]:
                sample = []
                sample.append(obs.state[env_i])
                sample.append(action[env_i])
                sample.append(obs.reward[env_i])
                sample.append(obs.done[env_i])
                sample.append(obs.is_first[env_i])
                sample.append(self.model_step.clone())
                buffer_infos = self.replay_buffer.append_step(sample, env_i)

                # Add Buffer Infos
                for key, value in buffer_infos.items():
                    self.add_info(key, value)

        ###############################################################################
        # Reset Env
        ###############################################################################

        # Is_last / Time Limit
        for env_i in range(self.config.num_envs):
            if obs.is_last[env_i]:

                # Set finished_episode
                finished_episode = []
                finished_episode.append(torch.stack(self.episode_history.episodes[env_i].states, dim=0))
                finished_episode.append(torch.stack(self.episode_history.episodes[env_i].actions, dim=0))
                finished_episode.append(torch.stack(self.episode_history.episodes[env_i].rewards, dim=0))
                finished_episode.append(torch.stack(self.episode_history.episodes[env_i].dones, dim=0))
                finished_episode.append(torch.stack(self.episode_history.episodes[env_i].is_firsts, dim=0))
                finished_episode.append(torch.stack(self.episode_history.episodes[env_i].model_steps, dim=0))

                # Copy Episode
                finished_episode = copy.deepcopy(finished_episode)

                # Add Infos
                self.add_info("episode_steps", self.episode_history.ep_step[env_i].item())
                self.add_info("episode_reward_total", self.ep_rewards[env_i].item())

                # Reset Episode Step
                self.episode_history.ep_step[env_i] = 0

                # Reset Hidden - Initialize with 2 timesteps for motion
                latent = self.rssm.initial(batch_size=1, seq_length=2, 
                                        dtype=torch.float32, detach_learned=True)
                action = torch.zeros(self.env.num_actions, dtype=torch.float32)
                self.episode_history.hidden[1][env_i] = action
                
                for key in self.episode_history.hidden[0]:
                    # Do not reset hidden
                    if key != "hidden":
                        # latent has shape (1, 2, ...), we need to extract it properly
                        self.episode_history.hidden[0][key][env_i] = latent[key].squeeze(dim=0)

                # Reset Env
                obs_reset = self.env.envs[env_i].reset()
                self.episode_history.state[env_i] = obs_reset.state

                # Reset Episode History
                self.episode_history.episodes[env_i] = AttrDict(
                    states=[obs_reset.state],
                    actions=[torch.zeros(self.env.num_actions, dtype=torch.float32)],
                    rewards=[obs_reset.reward],
                    dones=[obs_reset.done],
                    is_firsts=[obs_reset.is_first],
                    model_steps=[self.model_step.clone()]
                )

                # Update Traj Buffer
                sample = []
                sample.append(obs_reset.state)
                sample.append(torch.zeros(self.env.num_actions, dtype=torch.float32))
                sample.append(obs_reset.reward)
                sample.append(obs_reset.done)
                sample.append(obs_reset.is_first)
                sample.append(self.model_step.clone())
                buffer_infos = self.replay_buffer.append_step(sample, env_i)

                # Add Buffer Infos
                for key, value in buffer_infos.items():
                    self.add_info(key, value)

                # Update training_infos
                self.episodes += 1
                self.ep_rewards[env_i] = 0.0

        # Default Mode
        self.encoder_network.train(mode=training)
        self.rssm.train(mode=training)
        self.policy_network.train(mode=training)

    def update_target_networks(self):

        # Update Target Networks
        if 0 <= self.config.critic_ema_decay <= 1:

            # Soft Update
            for param_target, param_net in zip(self.v_target.parameters(), self.value_network.parameters()):
                param_target.mul_(1 - self.config.critic_ema_decay)
                param_target.add_(self.config.critic_ema_decay * param_net.detach())
        else:

            # Hard Update
            if self.model_step % self.config.critic_ema_decay == 0:
                self.v_target.load_state_dict(self.value_network.state_dict())

    def train_step(self, inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training):

        # Init Dict
        batch_losses = {}
        batch_metrics = {}

        # Preprocess state (uint8 to float32)
        inputs = self.preprocess_inputs(inputs, time_stacked=True)

        ###############################################################################
        # World Train Step
        ###############################################################################

        # World Model Step
        self.set_require_grad([self.policy_network, self.value_network], False)
        self.set_require_grad([self.encoder_network, self.decoder_network, self.rssm, self.reward_network, self.continue_network], True)
        world_model_batch_losses, world_model_batch_metrics, _ = self.world_model.train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)
        batch_losses.update({"world_model_" + key: value for key, value in world_model_batch_losses.items()})
        batch_metrics.update({"world_model_" + key: value for key, value in world_model_batch_metrics.items()})
        self.infos.update({"world_model_" + key: value for key, value in self.world_model.infos.items()})

        ###############################################################################
        # Actor Model Step
        ###############################################################################

        # Eval Mode: Disable Dropout
        self.rssm.eval()

        self.set_require_grad(self.policy_network, True)
        self.set_require_grad([self.value_network, self.encoder_network, self.decoder_network, self.rssm, self.reward_network, self.continue_network], False)
        actor_model_batch_losses, actor_model_batch_metrics, _ = self.actor_model.train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)
        batch_losses.update({"actor_model_" + key: value for key, value in actor_model_batch_losses.items()})
        batch_metrics.update({"actor_model_" + key: value for key, value in actor_model_batch_metrics.items()})
        self.infos.update({"actor_model_" + key: value for key, value in self.actor_model.infos.items()})

        ###############################################################################
        # Value Model Step
        ###############################################################################

        self.set_require_grad(self.value_network, True)
        self.set_require_grad([self.policy_network, self.encoder_network, self.decoder_network, self.rssm, self.reward_network, self.continue_network], False)
        critic_model_batch_losses, critic_model_batch_metrics, _ = self.critic_model.train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)
        batch_losses.update({"critic_model_" + key: value for key, value in critic_model_batch_losses.items()})
        batch_metrics.update({"critic_model_" + key: value for key, value in critic_model_batch_metrics.items()})
        self.infos.update({"critic_model_" + key: value for key, value in self.critic_model.infos.items()})

        # Train Mode
        self.rssm.train()

        ###############################################################################
        # Update Target Networks
        ###############################################################################

        # Update value target
        self.update_target_networks()

        ###############################################################################
        # Env Step
        ###############################################################################

        # Env Step
        num_env_steps = (self.config.batch_size * self.config.L) / (self.config.env_step_period * self.config.num_envs)

        # Env step every n model step
        if 0 < num_env_steps < 1:
            model_step_period = 1 / num_env_steps
            if self.model_step % model_step_period == 0:
                with torch.cuda.amp.autocast(enabled=precision!=torch.float32, dtype=precision):
                    self.env_step()
            
        # n env steps per model step
        else:
            with torch.cuda.amp.autocast(enabled=precision!=torch.float32, dtype=precision):
                for i in range(int(num_env_steps)):
                    self.env_step()

        # Update Infos
        self.infos["episodes"] = self.episodes.item()
        for env_i in range(self.config.num_envs):
            self.infos["ep_rewards_{}".format(env_i)] = round(self.ep_rewards[env_i].item(), 2)
        self.infos["step"] = self.model_step
        self.infos["action_step"] = self.action_step.item()

        # Built
        if not self.built:
            self.built = True

        return batch_losses, batch_metrics, _
    
    class WorldModel(models.Model):

        def __init__(self, outer):
            super().__init__(name="World Model")
            object.__setattr__(self, "outer", outer)
            self.encoder_network = self.outer.encoder_network
            self.decoder_network = self.outer.decoder_network
            self.continue_network = self.outer.continue_network
            self.reward_network = self.outer.reward_network
            self.rssm = self.outer.rssm
            # self.contrastive_network = self.outer.contrastive_network
            self.discriminator_network = self.outer.discriminator_network
            # self.temporal_order_discriminator = self.outer.temporal_order_discriminator

        def __getattr__(self, name):
            return getattr(self.outer, name)

        def forward(self, inputs):

            # Unpack Inputs 
            states, actions, rewards, dones, is_firsts, model_steps = inputs

            # Outputs
            outputs = {}

            ###############################################################################
            # Model Forward
            ###############################################################################

            assert actions.shape[1] == self.config.L
            B, L = actions.shape[:2]

            # Forward Representation Network (B, L, ...)
            latent = self.encoder_network(states)

            # Model Observe (B, L, D)
            posts, priors = self.rssm.observe(
                states=latent, 
                prev_actions=actions, 
                is_firsts=is_firsts, 
                prev_state=None, 
                is_firsts_hidden=None
            )

            # Update Hidden States
            is_firsts_hidden_concat = is_firsts

            # Get feat (B, L, Dfeat)
            feats = self.rssm.get_feat(posts)

            # Predict reward (B, L, 1)
            model_rewards = self.reward_network(feats)

            # Rec Images (B, L, ...)
            states_pred = self.decoder_network(posts["stoch"].flatten(-2, -1).detach() if self.config.detach_decoder else posts["stoch"].flatten(-2, -1))

            # Predict Discounts
            discount_pred = self.continue_network(feats)

            ###############################################################################
            # Model Temporal Loss
            # Now the code try to identify particular temporal orders in sequences of features
            # You can also make it realize if the window is consecutive or not
            ###############################################################################

            # # === 1. Extract real & fake features ===
            # real_feats = latent["stoch"].flatten(-2, -1).detach()     # (B, L, D) real, no grad
            # fake_feats = priors["stoch"].flatten(-2, -1)              # (B, L, D) fake, grad to RSSM

            # B, L, D = real_feats.shape

            # total_loss_tod_rssm = 0.0
            # active_tods = 0

            # for tod_idx, tod in enumerate(self.temporal_order_discriminator):
            #     window_size = self.config.window_size[tod_idx]

            #     if L < window_size:
            #         continue  # Skip TODs whose window size is too large

            #     active_tods += 1

            #     # === 2. Extract all consecutive windows ===
            #     num_windows = L - window_size + 1
            #     total_samples = B * num_windows

            #     # Build sliding windows efficiently (vectorized)
            #     windows_real = torch.stack(
            #         [real_feats[:, i:i+window_size] for i in range(num_windows)],
            #         dim=1
            #     ).reshape(total_samples, window_size, D)

            #     windows_fake = torch.stack(
            #         [fake_feats[:, i:i+window_size] for i in range(num_windows)],
            #         dim=1
            #     ).reshape(total_samples, window_size, D)

            #     # === 3. Random permutation assignment ===
            #     num_perms = tod.num_perm

            #     # Real windows
            #     perm_ids_real = torch.randint(0, num_perms, (total_samples,), device=real_feats.device)
            #     perm_real_idx = tod.perm_indices[perm_ids_real]              # (N, window_size)
            #     windows_perm_real = windows_real[torch.arange(total_samples).unsqueeze(-1), perm_real_idx]

            #     # Fake windows
            #     perm_ids_fake = torch.randint(0, num_perms, (total_samples,), device=fake_feats.device)
            #     perm_fake_idx = tod.perm_indices[perm_ids_fake]
            #     windows_perm_fake = windows_fake[torch.arange(total_samples).unsqueeze(-1), perm_fake_idx]

            #     # === 4. Train TOD on real windows ===
            #     logits_real = tod(windows_perm_real)
            #     loss_tod_real = F.cross_entropy(logits_real, perm_ids_real)

            #     # === Accuracy ===
            #     with torch.no_grad():
            #         preds = logits_real.argmax(dim=1)
            #         acc = (preds == perm_ids_real).float().mean().item()

            #     self.tod_optimizers[tod_idx].zero_grad(set_to_none=True)
            #     loss_tod_real.backward()
            #     self.tod_optimizers[tod_idx].step()

            #     print(f"TOD Window {window_size} - Acc: {acc*100:.2f}%, Loss TOD Real: {loss_tod_real.item():.4f}")

            #     # === 5. Train RSSM to fool TOD (TOD frozen) ===
            #     if self.model_step >= 20000:
            #         tod.eval()
            #         for p in tod.parameters():
            #             p.requires_grad = False

            #         logits_fake = tod(windows_perm_fake)
            #         loss_tod_rssm = F.cross_entropy(logits_fake, perm_ids_fake)

            #         for p in tod.parameters():
            #             p.requires_grad = True
            #         tod.train()
                    
            #         # print("logits_fake.grad_fn:", logits_fake.grad_fn)
            #         total_loss_tod_rssm += loss_tod_rssm


            # # === 6. Combine adversarial TOD loss into RSSM ===
            # if self.model_step >= 20000 and active_tods > 0:
            #     print("ahihi")
            #     avg_loss = total_loss_tod_rssm / active_tods
            #     self.add_loss("model_temporal_order", avg_loss, weight=self.config.tod_scale)

            ###############################################################################
            # Model One Discriminator Loss
            ###############################################################################

            # def discriminator_loss(D_real, D_fake):
            #     # real -> +1, fake -> -1
            #     loss_real = F.relu(1.0 - D_real).mean()
            #     loss_fake = F.relu(1.0 + D_fake).mean()
            #     return 0.5 * (loss_real + loss_fake)

            # def world_model_adv_loss(D_fake):
            #     # world model tries to make D_fake large (realistic)
            #     return -D_fake.mean()

            # # print(self.model_step)
            # if self.model_step >= 20000:
            #     real_latent = latent["stoch"].flatten(-2, -1).detach()
            #     fake_latent = priors["stoch"].flatten(-2, -1).detach()

            #     # Discriminator
            #     D_real = self.discriminator_network(real_latent)
            #     D_fake = self.discriminator_network(fake_latent)

            #     loss_D = discriminator_loss(D_real, D_fake)

            #     # Update discriminator
            #     self.discriminator_network.opt_disc.zero_grad(set_to_none=True)
            #     loss_D.backward()
            #     self.discriminator_network.opt_disc.step()

            #     D_fake_for_G = self.discriminator_network(priors["stoch"].flatten(-2, -1))
            #     self.add_loss("model_discriminator", world_model_adv_loss(D_fake_for_G), weight=0.3)
            
            ###############################################################################
            # Model Multiple Discriminator Losses
            ###############################################################################

            def discriminator_loss(D_real, D_fake):
                # real -> +1, fake -> -1
                loss_real = F.relu(1.0 - D_real).mean()
                loss_fake = F.relu(1.0 + D_fake).mean()
                return 0.5 * (loss_real + loss_fake)

            def world_model_adv_loss(D_fake):
                # world model tries to make D_fake large (realistic)
                return -D_fake.mean()

            if self.model_step >= 10000:
                real_latent = latent["stoch"].flatten(-2, -1).detach()  # [B, L, D]
                fake_latent = priors["stoch"].flatten(-2, -1).detach()  # [B, L, D]
                
                B, L, D = real_latent.shape
                
                total_loss_D = 0.0
                total_loss_G = 0.0
                num_discriminators = len(self.config.window_size)
                
                for disc_idx, (discriminator, optimizer, window_size, num_seq_to_discriminate) in enumerate(
                    zip(self.discriminator_network, self.discriminator_optimizers, self.config.window_size, self.config.num_seq_to_discriminate)
                ):
                    # Skip if sequence length is shorter than window size
                    if L < window_size:
                        continue
                    
                    # Number of possible windows per sequence
                    max_start = L - window_size
                    num_samples = min(num_seq_to_discriminate, max_start + 1)  # Can't sample more than available positions
                    
                    # Sample starting positions for each sequence in batch
                    # [B, num_samples]
                    start_positions = torch.stack([
                        torch.randperm(max_start + 1, device=real_latent.device)[:num_samples]
                        for _ in range(B)
                    ])
                    
                    # Extract windows: [B, num_samples, window_size, D]
                    real_windows = torch.stack([
                        torch.stack([
                            real_latent[b, start:start+window_size, :]
                            for start in start_positions[b]
                        ])
                        for b in range(B)
                    ])
                    
                    fake_windows = torch.stack([
                        torch.stack([
                            fake_latent[b, start:start+window_size, :]
                            for start in start_positions[b]
                        ])
                        for b in range(B)
                    ])
                    
                    # Reshape to [B*num_samples, window_size, D]
                    real_windows = real_windows.view(B * num_samples, window_size, D)
                    fake_windows = fake_windows.view(B * num_samples, window_size, D)
                    
                    # Forward pass through discriminator
                    D_real = discriminator(real_windows)  # [B*num_samples]
                    D_fake = discriminator(fake_windows)  # [B*num_samples]
                    
                    # Discriminator loss
                    loss_D = discriminator_loss(D_real, D_fake)
                    
                    # Update discriminator
                    optimizer.zero_grad(set_to_none=True)
                    loss_D.backward()
                    optimizer.step()
                    
                    total_loss_D += loss_D.item()
                    
                    if self.model_step >= 12000:
                        # Generator adversarial loss (using non-detached priors)
                        fake_windows_for_G = torch.stack([
                            torch.stack([
                                priors["stoch"].flatten(-2, -1)[b, start:start+window_size, :]
                                for start in start_positions[b]
                            ])
                            for b in range(B)
                        ]).view(B * num_samples, window_size, D)
                        
                        D_fake_for_G = discriminator(fake_windows_for_G)
                        loss_G = world_model_adv_loss(D_fake_for_G)
                        
                        total_loss_G += loss_G
                
                self.add_info("average_discriminator_loss", total_loss_D / num_discriminators)
                if self.model_step >= 12000 and num_discriminators > 0:
                    avg_loss_G = total_loss_G / num_discriminators
                    self.add_loss("discriminator", avg_loss_G, weight=0.3)
            ###############################################################################
            # Model Contrastive Loss
            ###############################################################################

            # # Flatten B and L to ensure different augmentation per sample (B*L, 3, H, W)
            # states_flatten = states.flatten(0, 1)

            # def _apply_aug(x):
            #     return self.config.contrastive_augments(x)

            # aug1_flat = torch.stack([_apply_aug(x) for x in states_flatten], dim=0)
            # aug2_flat = torch.stack([_apply_aug(x) for x in states_flatten], dim=0)

            # aug1 = aug1_flat.view(B, L, *states.shape[2:])
            # aug2 = aug2_flat.view(B, L, *states.shape[2:])

            # # === 2. Encode both augmented sequences ===
            # enc1 = self.encoder_network(aug1)
            # enc2 = self.encoder_network(aug2)

            # # === 3. Extract per-timestep features from RSSM ===
            # posts1, _ = self.rssm.observe(enc1, prev_actions=actions, is_firsts=is_firsts)
            # posts2, _ = self.rssm.observe(enc2, prev_actions=actions, is_firsts=is_firsts)
            # feats1 = self.rssm.get_feat(posts1)  # (B, L, D)
            # feats2 = self.rssm.get_feat(posts2)  # (B, L, D)

            # # === 4. Pass through contrastive network(s) ===
            # feats1, feats2 = self.contrastive_network[0](
            #     feats1=feats1,
            #     feats2=feats2,
            # )

            # # === 5. Compute temporal contrastive loss ===
            # info_nce_loss, acc_con = self.compute_temporal_contrastive_loss(
            #     feats1, feats2, window=self.config.get("contrastive_window", 0)
            # )

            # self.add_loss(
            #     name="model_contrastive",
            #     loss=info_nce_loss,
            #     weight=self.config.loss_contrastive_scale
            # )
            # self.add_metric("acc_con", acc_con)
            ###############################################################################
            # Model Reconstruction Loss
            ###############################################################################

            # Model Image Loss
            self.add_loss("model_image", - states_pred.log_prob(states.detach()).mean(), weight=self.config.loss_decoder_scale)

            ###############################################################################
            # Model kl Loss
            ###############################################################################

            # KL
            kl_prior = torch.distributions.kl.kl_divergence(self.rssm.get_dist({k: v if k == "hidden" else v.detach() for k, v in posts.items()}), self.rssm.get_dist(priors))
            kl_post = torch.distributions.kl.kl_divergence(self.rssm.get_dist(posts), self.rssm.get_dist({k: v if k == "hidden" else v.detach() for k, v in priors.items()}))

            # Add losses, Mean after Free Nats
            self.add_loss("kl_prior", torch.mean(torch.clip(kl_prior, min=self.config.free_nats)), weight=self.config.loss_kl_prior_scale)
            self.add_loss("kl_post", torch.mean(torch.clip(kl_post, min=self.config.free_nats)), weight=self.config.loss_kl_post_scale)

            ###############################################################################
            # Model Reward Loss
            ###############################################################################

            # Model Reward Loss
            self.add_loss("model_reward", - model_rewards.log_prob(rewards.unsqueeze(dim=-1).detach()).mean(), weight=self.config.loss_reward_scale)

            ###############################################################################
            # Model Discount Loss
            ###############################################################################

            # Model Discount Loss
            self.add_loss("model_discount", - discount_pred.log_prob((1.0 - dones).unsqueeze(dim=-1).detach()).mean(), self.config.loss_discount_scale)

            ###############################################################################
            # Flatten and Detach Posts
            ###############################################################################

            # Prepare stoch pairs for motion: (B, L, 2, stoch_size, discrete) or (B, L, 2, stoch_size)
            if self.config.discrete:
                # posts["stoch"] shape: (B, L, stoch_size, discrete)
                stoch_pairs = torch.stack([
                    # For t=0, use initial state as z_{t-1}
                    torch.cat([
                        self.rssm.initial(B, 1, posts["stoch"].dtype, posts["stoch"].device)["stoch"],
                        posts["stoch"][:, :-1]
                    ], dim=1),  # z_{t-1}: (B, L, stoch_size, discrete)
                    posts["stoch"]  # z_t: (B, L, stoch_size, discrete)
                ], dim=2)  # (B, L, 2, stoch_size, discrete)
            else:
                # posts["stoch"] shape: (B, L, stoch_size)
                stoch_pairs = torch.stack([
                    torch.cat([
                        self.rssm.initial(B, 1, posts["stoch"].dtype, posts["stoch"].device)["stoch"],
                        posts["stoch"][:, :-1]
                    ], dim=1),  # z_{t-1}: (B, L, stoch_size)
                    posts["stoch"]  # z_t: (B, L, stoch_size)
                ], dim=2)  # (B, L, 2, stoch_size)

            # K, V: (B, C+L, D) -> (B*L, C, D)
            hidden_flatten = [
                (
                    # Key (B*L, C, D)
                    torch.stack([
                        # Padd hidden if not enough left context (B, C, D)
                        torch.cat([
                            # Zero Padding to reach length (C,): max(0, L+C-1-t - len(h))
                            hidden_blk[0].new_zeros(
                                hidden_blk[0].shape[0], 
                                max(0, self.config.L + self.config.att_context_left - 1 - t - hidden_blk[0].shape[1]), 
                                hidden_blk[0].shape[2]
                            ), 
                            # hidden [-L+t+1 - C:-L+t+1]
                            hidden_blk[0][:, max(0, hidden_blk[0].shape[1] - self.config.L + t + 1 - self.config.att_context_left):
                                            hidden_blk[0].shape[1] - self.config.L + t + 1]
                        ], dim=1) 
                    for t in range(0, self.config.L)], dim=1).flatten(start_dim=0, end_dim=1).detach(), 

                    # Value (B*L, C, D)
                    torch.stack([
                        # Padd hidden if not enough left context (B, C, D)
                        torch.cat([
                            # zeros max(0, L+C-1-t - len(h))
                            hidden_blk[1].new_zeros(
                                hidden_blk[1].shape[0], 
                                max(0, self.config.L + self.config.att_context_left - 1 - t - hidden_blk[1].shape[1]), 
                                hidden_blk[1].shape[2]
                            ), 
                            # hidden [-L+t+1 - C:-L+t+1]
                            hidden_blk[1][:, max(0, hidden_blk[1].shape[1] - self.config.L + t + 1 - self.config.att_context_left):
                                            hidden_blk[1].shape[1] - self.config.L + t + 1]
                        ], dim=1) 
                    for t in range(0, self.config.L)], dim=1).flatten(start_dim=0, end_dim=1).detach(),
                )
                for hidden_blk in posts["hidden"]
            ]

            # is_firsts flatten (B, L) -> (B*L, 1), will result in masking hidden if true
            self.outer.detached_is_firsts = is_firsts.flatten(start_dim=0, end_dim=1).unsqueeze(dim=1).detach()

            # is_firsts hidden flatten (B, C+L) -> (B*L, C)
            self.outer.detached_is_firsts_hidden = torch.stack([
                torch.cat([
                    # Zero Padding to reach length (C,): max(0, L+C-1-t - len(h))
                    is_firsts_hidden_concat.new_zeros(
                        is_firsts_hidden_concat.shape[0], 
                        max(0, self.config.L + self.config.att_context_left - 1 - t - is_firsts_hidden_concat.shape[1])
                    ),  
                    # set first element to True in order to mask padding (1,)
                    is_firsts_hidden_concat.new_ones(is_firsts_hidden_concat.shape[0], 1),
                    # is_firsts [t-C + 1:t]
                    is_firsts_hidden_concat[:, max(0, is_firsts_hidden_concat.shape[1] - self.config.L + t + 1 - self.config.att_context_left):
                                            is_firsts_hidden_concat.shape[1] - self.config.L + t]
                ], dim=1) 
                for t in range(0, self.config.L)
            ], dim=1).flatten(start_dim=0, end_dim=1).detach()

            # Flatten and detach post with motion pairs (B, L, D) -> (B*L, 1 or 2, D) = (B', 1 or 2, D)
            self.outer.detached_posts = {
                k: hidden_flatten if k == "hidden" 
                else (
                    # For stoch, we need to flatten the pairs: (B, L, 2, ...) -> (B*L, 2, ...)
                    v.flatten(start_dim=0, end_dim=1).detach() if k == "stoch" and len(v.shape) > 3
                    else v.flatten(start_dim=0, end_dim=1).unsqueeze(dim=1).detach()
                )
                for k, v in {**posts, "stoch": stoch_pairs}.items()
            }
            return outputs
        
    class ActorModel(models.Model):

        def __init__(self, outer):
            super().__init__(name="Actor Model")
            object.__setattr__(self, "outer", outer)
            self.policy_network = self.outer.policy_network

        def __getattr__(self, name):
            return getattr(self.outer, name)
        
        def forward(self, inputs):

            # Unpack Inputs 
            states, actions, rewards, dones, is_firsts, model_steps  = inputs

            # Outputs
            outputs = {}

            ###############################################################################
            # Policy Forward
            ###############################################################################

            prev_state = self.detached_posts

            # Model Imagine H next states (B', 1+H, D) with state synchronized actions
            img_states = self.rssm.imagine(
                p_net=self.policy_network, 
                prev_state=prev_state, 
                img_steps=self.config.H,
                is_firsts=self.detached_is_firsts,
                is_firsts_hidden=self.detached_is_firsts_hidden
            )

            # Get feat (B', 1+H, Dfeat)
            feats = self.rssm.get_feat(img_states)

            # Predict rewards (B', 1+H, 1)
            model_rewards = self.reward_network(feats)

            # Predict Values (B', 1+H, 1)
            if self.config.target_value_reg:
                values = self.value_network(feats)
            else:
                values = self.v_target(feats)

            # Predict Discounts (B', 1+H, 1)
            discounts = self.continue_network(feats).mode # 0 / 1

            # Override discount prediction for the first step with the true
            # discount factor from the replay buffer.
            true_first = (1.0 - dones.flatten(start_dim=0, end_dim=1)).unsqueeze(dim=-1).unsqueeze(dim=-1) # 0 or 1
            discounts = torch.cat([true_first, discounts[:, 1:]], dim=1)

            ###############################################################################
            # Policy Loss
            ###############################################################################

            # (B', 1+H, 1)
            weights = torch.cumprod(self.config.gamma * discounts, dim=1).detach() / self.config.gamma

            # Compute lambda returns (B', H, 1), one action grad lost because of next value
            returns = self.compute_td_lambda(rewards=model_rewards.mode()[:, 1:], values=values.mode()[:, 1:], discounts=self.config.gamma * discounts[:, 1:])
            self.add_info("returns_mean", returns.mean().item())

            # Update Perc
            offset, invscale = self.update_perc(returns)

            # Norm Returns using quantiles ema ~ [0:1]
            normed_returns = (returns - offset) / invscale # 1:H+1
            normed_base = (values.mode()[:, :-1] - offset) / invscale # 0:H

            # advantage (B', H)
            advantage = (normed_returns - normed_base).squeeze(dim=-1)

            # Policy Dist (B', 1+H, A)
            policy_dist = self.policy_network(feats.detach()) 

            # Actor Loss
            if self.config.actor_grad == "dynamics":
                actor_loss = advantage
            elif self.config.actor_grad == "reinforce":
                actor_loss = policy_dist.log_prob(img_states["action"].detach())[:, :-1] * advantage.detach()
            else:
                raise Exception("Unknown actor grad: {}".format(self.actor_grad))
            
            # Add Negative Entropy loss
            policy_ent = policy_dist.entropy()[:, :-1]
            self.add_info("policy_ent", policy_ent.mean().item())
            actor_loss += self.config.eta_entropy * policy_ent

            # Apply weights
            actor_loss *= weights[:, :-1].squeeze(dim=-1)

            # Add loss
            self.add_loss("actor", - actor_loss.mean())  

            self.outer.detached_feats = feats.detach()
            self.outer.detached_returns = returns.detach()
            self.outer.detached_weights = weights.detach()

            return outputs
        
    class CriticModel(models.Model):

        def __init__(self, outer):
            super().__init__(name="Critic Model")
            object.__setattr__(self, "outer", outer)
            self.value_network = self.outer.value_network

        def __getattr__(self, name):
            return getattr(self.outer, name)
        
        def forward(self, inputs):

            # Unpack Inputs 
            states, actions, rewards, dones, is_firsts, model_steps  = inputs

            # Outputs
            outputs = {}

            ###############################################################################
            # Value Loss
            ###############################################################################

            feats = self.detached_feats
            returns = self.detached_returns
            weights = self.detached_weights

            # Value (B', H, 1)
            value_dist = self.value_network(feats.detach()[:, :-1])

            # Value Loss
            value_loss = value_dist.log_prob(returns.detach())
            
            # Add Regularization
            if self.config.target_value_reg:
                with torch.no_grad():
                    value_target = self.v_target(feats.detach()[:, :-1]).mode()
                value_loss += self.config.critic_slow_reg_scale * value_dist.log_prob(value_target.detach())

            # Weight loss
            value_loss *= weights[:, :-1].squeeze(dim=-1)

            # Add Loss
            self.add_loss("value", - value_loss.mean())

            return outputs
    
    def update_perc(self, returns):

        # Compute percentiles (,)
        low = torch.quantile(returns.detach(), q=self.config.return_norm_perc_low)
        high = torch.quantile(returns.detach(), q=self.config.return_norm_perc_high)

        # Update percentiles ema
        self.perc_low = self.config.return_norm_decay * self.perc_low + (1 - self.config.return_norm_decay) * low
        self.perc_high = self.config.return_norm_decay * self.perc_high + (1 - self.config.return_norm_decay) * high
        self.add_info("perc_low", self.perc_low.item())
        self.add_info("perc_high", self.perc_high.item())

        # Compute offset, invscale
        offset = self.perc_low
        invscale = torch.clip(self.perc_high - self.perc_low, min=1.0 / self.config.return_norm_limit)

        return offset.detach(), invscale.detach()
    
    def get_perc(self):

        # Compute offset, invscale
        offset = self.perc_low
        invscale = torch.clip(self.perc_high - self.perc_low, min=1.0 / self.config.return_norm_limit)

        return offset.detach(), invscale.detach()
    
    def compute_td_lambda(self, rewards, values, discounts):

        # Init for loop
        interm = rewards + discounts * (1 - self.config.lambda_td) * values
        vals = [values[:, -1]]

        # Recurrence loop
        for t in reversed(range(interm.shape[1])):
            vals.append(interm[:, t] + discounts[:, t] * self.config.lambda_td * vals[-1])

        # Stack and slice init val
        lambda_values = torch.stack(list(reversed(vals))[:-1], dim=1)

        return lambda_values
    
    def compute_temporal_contrastive_loss(self, feats_view1, feats_view2, window=1):
        """
        Contrastive loss between two augmented sequences.

        feats_view1, feats_view2: [B, L, D]
            Latent features from the two augmented views.
        window: int, temporal neighborhood for positives (±window)
        """
        B, L, D = feats_view1.shape

        # Flatten
        anchors = feats_view1.reshape(B * L, D)
        feats_all = feats_view2.reshape(B * L, D)

        # Normalize (cosine sim)
        anchors = F.normalize(anchors, dim=-1)
        feats_all = F.normalize(feats_all, dim=-1)

        # Compute similarity matrix
        sim_matrix = anchors @ feats_all.T  # (B*L, B*L)

        # Scale by temperature
        temp = self.config.get("contrastive_temp", 0.1)
        sim_matrix = sim_matrix / temp

        # Build positive mask
        pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)

        for b in range(B):
            for t in range(L):
                anchor_idx = b * L + t
                for dt in range(-window, window + 1):
                    tp = t + dt
                    if 0 <= tp < L:
                        pos_mask[anchor_idx, b * L + tp] = True

        # InfoNCE: multi-positive version
        logsumexp_all = torch.logsumexp(sim_matrix, dim=-1)  # (B*L,)
        losses = []
        for i in range(B * L):
            pos_sims = sim_matrix[i][pos_mask[i]]
            if pos_sims.numel() == 0:
                continue
            logsumexp_pos = torch.logsumexp(pos_sims, dim=-1)
            losses.append(-(logsumexp_pos - logsumexp_all[i]))
        loss = torch.stack(losses).mean()

        # Accuracy: check if max similarity corresponds to any positive
        preds = sim_matrix.argmax(dim=-1)
        correct_mask = torch.zeros_like(preds, dtype=torch.bool)
        for b in range(B):
            for t in range(L):
                anchor_idx = b * L + t
                for dt in range(-window, window + 1):
                    tp = t + dt
                    if 0 <= tp < L and preds[anchor_idx] == b * L + tp:
                        correct_mask[anchor_idx] = True
                        break
        acc = correct_mask.float().mean()

        return loss, acc
    
    def compute_contrastive_loss(self, features_x, features_y):

        # Flatten (B', D)
        features_x = features_x.flatten(start_dim=0, end_dim=1)
        features_y = features_y.flatten(start_dim=0, end_dim=1)

        # Matmul (B', B')
        features = features_x.matmul(features_y.transpose(0, 1))

        # Diag (B',)
        features_pos = torch.diag(features)

        # Exp -> Sum -> Log: (B',)
        features_all = torch.logsumexp(features, dim=-1)
                        
        # Info NCE Loss: (B',)
        info_nce_loss = features_pos - features_all

        # Accuracy Contrastive
        acc_con = torch.mean(torch.where(features.argmax(dim=-1).cpu() == torch.arange(0, features.shape[0]), 1.0, 0.0))

        return info_nce_loss, acc_con

    def play(self, verbose=False, return_att_w=False):

        # Reset
        obs = self.env_eval.reset()

        # Transfer to device
        state = self.transfer_to_device(obs.state)
        prev_latent = self.transfer_to_device(self.rssm.initial(batch_size=1, seq_length=2, dtype=obs.reward.dtype, detach_learned=True))
        prev_action = self.transfer_to_device(torch.zeros(1, self.env.num_actions, dtype=obs.reward.dtype))

        # Create hidden
        hidden = (prev_latent, prev_action)

        # Init values
        total_rewards = 0
        step = 0

        # att weights
        if return_att_w:
            att_ws = []

        # Episode loop
        while 1:

            # Unpack hidden
            prev_latent, prev_action = hidden

            # Representation Network
            with torch.no_grad():

                # Repr State (1, ...)
                latent = self.encoder_network(self.preprocess_inputs(state.unsqueeze(dim=0), time_stacked=False))

                # Unsqueeze Time dim (B, 1, ...)
                latent = {key: value.unsqueeze(dim=1) for key, value in latent.items()}

                # RSSM (B, 1, ...)
                latent, _ = self.rssm(
                    states=latent, 
                    prev_states=prev_latent,
                    prev_actions=prev_action.unsqueeze(dim=1), 
                    is_firsts=torch.zeros(1, 1),
                    return_att_w=return_att_w
                )

                # Get feat (B, Dfeat)
                feat = self.rssm.get_feat(latent).squeeze(dim=1)

                # att weights
                if return_att_w:
                    att_ws.append(latent["att_w"])

                # Policy
                action = self.policy_network(feat).mode()

            # Update Hidden - maintain 2-timestep window for motion
            # Shift window: drop oldest, keep recent, add new
            latent_for_hidden = {}
            for key in latent.keys():
                if key == "hidden":
                    latent_for_hidden[key] = self.rssm.slice_hidden(latent[key])
                else:
                    # Create 2-timestep window: [prev_latent's recent, new latent]
                    # prev_latent has shape (B, 2, ...), we take index 1 (most recent)
                    # new latent has shape (B, 1, ...), we concatenate them
                    latent_for_hidden[key] = torch.cat([
                        prev_latent[key][:, -1:],  # Most recent from previous
                        latent[key]  # New latent
                    ], dim=1)  # Result: (B, 2, ...)
            
            hidden = (latent_for_hidden, action)

            # Forward Env
            obs = self.env_eval.step(action.argmax(dim=-1).squeeze(dim=0) if self.config.policy_discrete else action.squeeze(dim=0))
            state = self.transfer_to_device(obs.state)
            step += self.env_eval.action_repeat
            total_rewards += obs.reward

            # Done / Time Limit
            if obs.done or step >= self.config.time_limit_eval:
                break

        outputs = AttrDict({"score": total_rewards, "steps": step})
        if return_att_w:
            outputs.att_w = att_ws

        return outputs

    def eval_step(self, inputs, targets, verbose=False):

        # play
        outputs_ = self.play(verbose=verbose)
        outputs = {"score": torch.tensor(outputs_.score), "steps": torch.tensor(outputs_.steps)}

        # Update Infos
        for key, value in outputs.items():
            self.infos["ep_{}".format(key)] = value.item()

        # batch_losses, batch_metrics, batch_truths, batch_preds
        return {}, outputs, {}, {}
    
    def log_figure(self, step, inputs, targets, writer, tag, save_image=False):
        # Eval Mode
        mode = self.training
        self.eval()

        # Preprocess state (uint8 to float32)
        inputs = self.preprocess_inputs(inputs, time_stacked=True)

        # Unpack Inputs 
        states, actions, rewards, dones, is_firsts, model_steps = inputs

        # Number of Rows
        states = states[:self.config.log_figure_batch]
        actions = actions[:self.config.log_figure_batch]
        is_firsts = is_firsts[:self.config.log_figure_batch]

        with torch.no_grad():
            # Forward Representation Network (B, L, D)
            latent = self.encoder_network(states)

            ###############################################################################
            # Model
            ###############################################################################

            # Model Observe (B, L, D)
            posts, priors = self.rssm.observe(
                states=latent,
                prev_actions=actions,
                is_firsts=is_firsts,
                prev_state=None,
                is_firsts_hidden=None,
            )

            # Get feat (B, L, 2*D)
            feats = self.rssm.get_feat(posts)

            # Rec States (B, L, ...)
            states_rec = self.decoder_network(posts["stoch"].flatten(-2, -1)).mode()

            ###############################################################################
            # Imaginary (modified for 2-timestep motion computation)
            ###############################################################################

            # Initial State
            if self.config.log_figure_context_frames == 0:
                # No context: Need 2 initial states for motion computation
                initial_state = self.transfer_to_device(
                    self.rssm.initial(batch_size=feats.shape[0], seq_length=2, dtype=feats.dtype)
                )
                prev_state = initial_state
            elif self.config.log_figure_context_frames == 1:
                # Only 1 context frame: duplicate it to create 2-timestep window
                hidden_len = self.rssm.get_hidden_len(posts["hidden"])
                prev_state = {
                    k: [
                        (
                            v_blk[0][:, max(0, hidden_len - self.config.L):hidden_len - self.config.L + 1],
                            v_blk[1][:, max(0, hidden_len - self.config.L):hidden_len - self.config.L + 1]
                        ) for v_blk in v
                    ] if k == "hidden" else (
                        v[:, 0:1].repeat(1, 2, *([1] * (v.dim() - 2))) if k == "stoch" 
                        else v[:, 0:1]
                    ) for k, v in posts.items()
                }
            else:
                # 2+ context frames: use last 2 frames for motion computation
                context_start = self.config.log_figure_context_frames - 2
                context_end = self.config.log_figure_context_frames
                hidden_len = self.rssm.get_hidden_len(posts["hidden"])
                
                prev_state = {
                    k: [
                        (
                            v_blk[0][:, max(0, hidden_len - self.config.L + context_start - self.config.att_context_left):hidden_len - self.config.L + context_end],
                            v_blk[1][:, max(0, hidden_len - self.config.L + context_start - self.config.att_context_left):hidden_len - self.config.L + context_end]
                        ) for v_blk in v
                    ] if k == "hidden" else (
                        v[:, context_start:context_end] if k == "stoch"
                        else v[:, context_end - 1:context_end]
                    ) for k, v in posts.items()
                }

            # Model Imagine (B, 1+L-C, D)
            img_states = self.rssm.imagine(
                p_net=self.policy_network,
                prev_state=prev_state,
                img_steps=self.config.L - self.config.log_figure_context_frames,
                is_firsts=None,
                is_firsts_hidden=None
            )

            # Img States (B, L, ...)
            # Note: img_states["stoch"] now starts with the most recent context state
            if self.config.log_figure_context_frames == 0:
                # No context: use all imagined states (skip the initial duplicated state)
                states_img = self.decoder_network(
                    img_states["stoch"][:, 1:].flatten(-2, -1)
                ).mode()
            else:
                # With context: combine context frames with imagined states
                states_img = self.decoder_network(
                    torch.cat([
                        posts["stoch"][:, :self.config.log_figure_context_frames].flatten(-2, -1),
                        img_states["stoch"][:, 1:].flatten(-2, -1)
                    ], dim=1)
                ).mode()

        # Shift to 0..1 for display
        states_shift = states.clip(-0.5, 0.5) + 0.5
        states_rec_shift = states_rec.clip(-0.5, 0.5) + 0.5
        error_shift = 1 - torch.abs(states_rec_shift - states_shift).mean(dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
        states_img_shift = states_img.clip(-0.5, 0.5) + 0.5

        # Expand is_firsts
        is_firsts = is_firsts.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1).expand_as(states) * states_shift

        # Concat Outputs
        outputs = torch.concat([
            is_firsts,
            states_shift,
            states_rec_shift,
            error_shift,
            states_img_shift,
        ], dim=1).flatten(start_dim=0, end_dim=1)

        # Add Figure to logs
        if writer is not None:
            # Log Image (main)
            fig = torchvision.utils.make_grid(outputs, nrow=self.config.L, normalize=False, scale_each=False).cpu()
            writer.add_image(tag, fig, step)

        # Default Mode: restore training/eval mode
        self.train(mode=mode)