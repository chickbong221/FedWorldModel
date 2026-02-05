    def set_replay_buffer(self, replay_buffer):
        """
        Updated to initialize with 2 timesteps for motion computation
        """
        # Replay Buffer
        self.replay_buffer = replay_buffer

        # Set History - Initialize with 2 timesteps for motion
        obs_reset = self.env.reset()
        self.episode_history = AttrDict(
            ep_step=torch.zeros(self.config.num_envs),  # (N,)
            # Changed: Initialize with seq_length=2 for motion computation
            hidden=(
                self.rssm.initial(batch_size=self.config.num_envs, seq_length=2, 
                                dtype=torch.float32, detach_learned=True), 
                torch.zeros(self.config.num_envs, self.env.num_actions, dtype=torch.float32)
            ), 
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

    def env_step(self):
        """
        Updated to handle 2-timestep latent states for motion computation
        """
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

        def forward(
            self,
            states,
            prev_states,
            prev_actions,
            is_firsts,
            is_firsts_hidden=None,
            return_att_w=False,
            return_blocks_deter=False,
        ):
            """
            Motion-based forward pass using ONLY previous states.
            prev_states must contain at least 2 timesteps: (B, L+1, ...)
            """

            # ------------------------------------------------------------
            # Shape checks
            # ------------------------------------------------------------
            assert prev_actions.dim() == 3          # (B, L, A)
            assert is_firsts.dim() == 2              # (B, L)

            B, L, _ = prev_actions.shape

            # ------------------------------------------------------------
            # Action clipping
            # ------------------------------------------------------------
            if self.action_clip > 0.0:
                prev_actions *= (
                    self.action_clip
                    / torch.clip(torch.abs(prev_actions), min=self.action_clip)
                ).detach()

            # ------------------------------------------------------------
            # Attention mask
            # ------------------------------------------------------------
            mask = modules.return_mask(
                seq_len=L,
                hidden_len=self.get_hidden_len(prev_states["hidden"]),
                left_context=self.att_context_left,
                right_context=0,
                dtype=prev_actions.dtype,
                device=prev_actions.device,
            )

            # ------------------------------------------------------------
            # Handle resets
            # ------------------------------------------------------------
            if is_firsts.any():
                is_firsts = is_firsts.unsqueeze(-1)  # (B, L, 1)

                # Reset actions
                prev_actions *= (1.0 - is_firsts)

                # Reset states except hidden
                init_state = self.initial(
                    batch_size=B,
                    seq_length=L + 1,   # prev_states include extra step
                    dtype=prev_actions.dtype,
                    device=prev_actions.device,
                )

                for key, value in prev_states.items():
                    if key == "hidden":
                        continue
                    is_firsts_r = is_firsts.view(
                        is_firsts.shape + (1,) * (value.dim() - is_firsts.dim())
                    )
                    prev_states[key][:, 1:] = (
                        value[:, 1:] * (1.0 - is_firsts_r)
                        + init_state[key][:, 1:] * is_firsts_r
                    )

                is_firsts_mask = modules.return_is_firsts_mask(
                    is_firsts.squeeze(-1),
                    is_firsts_hidden=is_firsts_hidden,
                )
                mask = mask.minimum(is_firsts_mask)

            # ------------------------------------------------------------
            # Extract z_{t-2} and z_{t-1}
            # ------------------------------------------------------------
            # prev_states["stoch"]: (B, L+1, ...)
            stoch_tm2 = prev_states["stoch"][:, :-1]   # z_{t-2}
            stoch_tm1 = prev_states["stoch"][:, 1:]    # z_{t-1}

            if self.discrete:
                stoch_tm2 = stoch_tm2.flatten(start_dim=-2, end_dim=-1)
                stoch_tm1 = stoch_tm1.flatten(start_dim=-2, end_dim=-1)

            # ------------------------------------------------------------
            # Motion computation (NO current state)
            # ------------------------------------------------------------
            motion = self.compute_motion(stoch_tm1, stoch_tm2)

            # ------------------------------------------------------------
            # Motion + action
            # ------------------------------------------------------------
            motion_action = self.motion_action_mixer(
                torch.cat([motion, prev_actions], dim=-1)
            )

            # ------------------------------------------------------------
            # Transformer dynamics
            # ------------------------------------------------------------
            outputs = self.transformer(
                motion_action,
                hidden=prev_states["hidden"],
                mask=mask,
                return_hidden=True,
                return_att_w=return_att_w,
                return_blocks_x=return_blocks_deter,
            )

            m_t, hidden = outputs.x, outputs.hidden

            # ------------------------------------------------------------
            # Optional outputs
            # ------------------------------------------------------------
            add_out = {}
            if return_att_w:
                add_out["att_w"] = outputs.att_w
            if return_blocks_deter:
                add_out["blocks_deter"] = outputs.blocks_x

            # ------------------------------------------------------------
            # Fuse with z_{t-1}
            # ------------------------------------------------------------
            fused = self.dynamics_fusion(
                torch.cat([m_t, stoch_tm1], dim=-1)
            )

            # ------------------------------------------------------------
            # Prior distribution
            # ------------------------------------------------------------
            logits_prior = self.dynamics_predictor(fused).reshape(
                fused.shape[:-1] + (self.stoch_size, self.discrete)
            )

            prior = {
                "stoch": self.get_dist({"logits": logits_prior}).rsample(),
                "deter": fused,
                "hidden": hidden,
                "logits": logits_prior,
                **add_out,
            }

            # ------------------------------------------------------------
            # Posterior (uses current observation)
            # ------------------------------------------------------------
            post = self.forward_obs(fused, hidden, states)

            if return_att_w:
                post["att_w"] = prior["att_w"]
            if return_blocks_deter:
                post["blocks_deter"] = prior["blocks_deter"]

            return post, prior