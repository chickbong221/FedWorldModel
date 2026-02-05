        # Motion Extractor
        self.motion_type = motion_type
        self.freq_keep_ratio = freq_keep_ratio

        # Motion feature dimensionality
        if self.motion_type == "difference":
            motion_dim = self.stoch_size * self.discrete if self.discrete else self.stoch_size

        elif self.motion_type == "correlation":
            base_dim = self.stoch_size * self.discrete if self.discrete else self.stoch_size
            motion_dim = base_dim * base_dim   # cost volume flattened

        elif self.motion_type == "frequency":
            base_dim = self.stoch_size * self.discrete if self.discrete else self.stoch_size
            motion_dim = base_dim               # low+high concatenation preserves dim

        else:
            raise ValueError(f"Unknown motion type: {self.motion_type}")

        # Project motion + action to hidden_size for transformer
        self.motion_action_mixer = modules.MultiLayerPerceptron(
            dim_input=motion_dim + self.num_actions,
            dim_layers=[self.hidden_size, self.hidden_size],
            act_fun=[self.act_fun, None],
            weight_init=self.weight_init,
            bias_init=self.bias_init,
            norm=[self.norm, None] if module_pre_norm else self.norm,
            bias=self.norm is None
        )

        # Transformer processes motion: m_t
        self.transformer = modules.TransformerNetwork(
            dim_model=self.hidden_size,
            num_blocks=self.num_blocks,
            att_params={
                "class": "RelPosMultiHeadSelfAttention",
                "params": {
                    "num_heads": self.num_heads, 
                    "weight_init": "default", 
                    "bias_init": "default", 
                    "attn_drop_rate": self.drop_rate, 
                    "max_pos_encoding": self.max_pos_encoding, 
                    "causal": True
                }
            },
            emb_drop_rate=0.0,
            drop_rate=self.drop_rate,
            pos_embedding=None,
            mask=None,
            ff_ratio=self.ff_ratio,
            weight_init="default", 
            bias_init="default",
            act_fun="ReLU",
            module_pre_norm=module_pre_norm
        )

        # Fuse motion transformer output (m_t) with previous stochastic latent (z_{t-1})
        self.dynamics_fusion = modules.MultiLayerPerceptron(
            dim_input=self.hidden_size + (self.stoch_size * self.discrete if self.discrete else self.stoch_size),
            dim_layers=[
                self.hidden_size * 2,   # expansion
                self.hidden_size        # bottleneck
            ],
            act_fun=[
                self.act_fun,
                self.act_fun
            ],
            weight_init=self.weight_init,
            bias_init=self.bias_init,
            norm=self.norm
        )

        # Dynamics Predictor: fused features -> z_t distribution
        self.dynamics_predictor = modules.Linear(
            in_features=self.hidden_size, 
            out_features=self.discrete * self.stoch_size if self.discrete else 2 * self.stoch_size,
            weight_init=self.dist_weight_init,
            bias_init=self.dist_bias_init
        )

        if self.learn_initial:
            self.weight_init = nn.Parameter(torch.zeros(self.hidden_size))

    def motion_difference(self, z_t, z_tm1):
        """
        z_t, z_tm1: (B, L, S, D) or (B, L, S)
        """
        return z_t - z_tm1

    def motion_correlation(self, z_t, z_tm1):
        """
        Computes pairwise dot-product cost volume and compresses it.
        """

        # Flatten stochastic dims
        if self.discrete:
            z_t = z_t.flatten(start_dim=-2)      # (B, L, D)
            z_tm1 = z_tm1.flatten(start_dim=-2)

        # Normalize for stability
        z_t = torch.nn.functional.normalize(z_t, dim=-1)
        z_tm1 = torch.nn.functional.normalize(z_tm1, dim=-1)

        # Cost volume: (B, L, D, D)
        cost = torch.einsum("bld,blc->bldc", z_t, z_tm1) # May need efficient correlation

        # Compress cost volume
        cost = cost.reshape(cost.shape[0], cost.shape[1], -1)

        return cost
    
    def motion_frequency(self, z_t, z_tm1):
        """
        Frequency decomposition over latent dimension.
        """

        if self.discrete:
            z_t = z_t.flatten(start_dim=-2)
            z_tm1 = z_tm1.flatten(start_dim=-2)

        # FFT over latent dimension
        fft_t = torch.fft.fft(z_t, dim=-1)
        fft_tm1 = torch.fft.fft(z_tm1, dim=-1)

        # Frequency magnitude
        freq_mag = torch.abs(fft_t - fft_tm1)

        # Split low / high frequency bands
        D = freq_mag.shape[-1]
        k = int(D * self.freq_keep_ratio)

        low_freq = freq_mag[..., :k]
        high_freq = freq_mag[..., k:]

        # Concatenate bands
        return torch.cat([low_freq, high_freq], dim=-1)
    
    def compute_motion(self, z_t, z_tm1):

        if self.motion_type == "difference":
            motion = self.motion_difference(z_t, z_tm1)

        elif self.motion_type == "correlation":
            motion = self.motion_correlation(z_t, z_tm1)

        elif self.motion_type == "frequency":
            motion = self.motion_frequency(z_t, z_tm1)

        else:
            raise ValueError(f"Unknown motion type: {self.motion_type}")

        return motion

    def get_stoch(self, deter):
        
        # Linear Logits
        logits = self.dynamics_predictor(deter).reshape(deter.shape[:-1] + (self.stoch_size, self.discrete))
        dist_params = {'logits': logits}
    
        # Get Mode
        stoch = self.get_dist(dist_params).mode()

        return stoch

    def initial(self, batch_size=1, seq_length=1, dtype=torch.float32, device="cpu", detach_learned=False):

        initial_state = structs.AttrDict(
            logits=torch.zeros(batch_size, seq_length, self.stoch_size, self.discrete, dtype=dtype, device=device),
            stoch=torch.zeros(batch_size, seq_length, self.stoch_size, self.discrete, dtype=dtype, device=device),
            deter=torch.zeros(batch_size, seq_length, self.hidden_size, dtype=dtype, device=device),
            hidden=None
        )

        # Learned Initial
        if self.learn_initial:
            initial_state.deter = self.weight_init.repeat(batch_size, seq_length, 1)
            initial_state.stoch = self.get_stoch(initial_state.deter) 

            # Detach Learned
            if detach_learned:
                initial_state.deter = initial_state.deter.detach()
                initial_state.stoch = initial_state.stoch.detach()

        return initial_state

    def observe(self, states, prev_actions, is_firsts, prev_state=None, is_firsts_hidden=None, 
            return_blocks_deter=False):
        """
        Modified observe to initialize with 2 timesteps instead of 1.
        """
        # Create prev_states (B, L-1, ...)
        prev_states = {key: value[:, :-1] for key, value in states.items()}

        # Initial State - now creates 2 timesteps
        if prev_state is None:
            prev_actions[:, 0] = 0.0
            # Initialize with 2 timesteps for motion computation
            prev_state = self.initial(batch_size=prev_actions.shape[0], seq_length=2, 
                                    dtype=prev_actions.dtype, device=prev_actions.device)
            is_firsts_hidden = None

        # Concat prev_state (B, L, ...)
        # For most keys, concatenate along time dimension
        prev_states = {key: torch.cat([prev_state[key], value], dim=1) for key, value in prev_states.items()}
        # Hidden is handled separately
        prev_states["hidden"] = prev_state["hidden"]

        # Forward Model (B, L, D)
        posts, priors = self(states, prev_states, prev_actions, is_firsts, is_firsts_hidden, 
                            return_blocks_deter=return_blocks_deter)

        return posts, priors

    def imagine(self, p_net, prev_state, img_steps=1, is_firsts=None, is_firsts_hidden=None, actions=None):
        """
        Modified imagine to handle motion computation with 2-timestep prev_state.
        Expects prev_state["stoch"] to have shape (B, 2, ...) for motion computation.
        """
        # Policy
        policy = lambda s: p_net(self.get_feat(s).detach()).rsample()
        
        # Current state action - use the most recent state (index 1)
        if actions is None:
            # For policy sampling, we need a single state, use the most recent one
            single_state = {k: v[:, -1:] if k != "hidden" else v for k, v in prev_state.items()}
            prev_state["action"] = policy(single_state)
        else:
            assert actions.shape[1] == img_steps
            prev_state["action"] = actions[:, :1]

        # Initialize imagination states
        # Use the most recent stoch state for recording
        img_states = {
            "stoch": [prev_state["stoch"][:, -1:]], 
            "deter": [prev_state["deter"][:, -1:]] if "deter" in prev_state else [], 
            "logits": [prev_state["logits"][:, -1:]] if "logits" in prev_state else [], 
            "action": [prev_state["action"]]
        }
        
        # Model Recurrent loop with St, At
        for h in range(img_steps):
            # Compute mask
            mask = modules.return_mask(
                seq_len=1, 
                hidden_len=self.get_hidden_len(prev_state["hidden"]), 
                left_context=self.att_context_left, 
                right_context=0, 
                dtype=prev_state["action"].dtype, 
                device=prev_state["action"].device
            )
            
            if is_firsts_hidden is not None:
                # Append is_first mask
                is_firts_mask = modules.return_is_firsts_mask(is_firsts=is_firsts, 
                                                            is_firsts_hidden=is_firsts_hidden)
                mask = mask.minimum(is_firts_mask)

                # Concat is_firsts to hidden is_firsts (B, C)
                is_firsts_hidden = torch.cat([is_firsts_hidden[:, 1:], is_firsts], dim=1)
                
                # Set is_firsts to zero (B, 1)
                is_firsts = torch.zeros_like(is_firsts)
            
            # Forward Model - prev_state already contains both timesteps
            img_state = self.forward_img(
                prev_states=prev_state, 
                prev_actions=prev_state["action"], 
                mask=mask,
            )
            
            # Current state action
            if actions is None or h == img_steps - 1:
                img_state["action"] = policy(img_state)
            else:
                img_state["action"] = actions[:, h+1:h+2]

            # Slice hidden
            img_state["hidden"] = self.slice_hidden(img_state["hidden"])

            # Update previous state for next iteration
            # Shift the 2-timestep window: drop oldest, keep recent, add new
            prev_state = {
                "stoch": torch.cat([prev_state["stoch"][:, -1:], img_state["stoch"]], dim=1),
                "deter": img_state["deter"],
                "hidden": img_state["hidden"],
                "logits": img_state["logits"],
                "action": img_state["action"]
            }

            # Append to Lists (only the new state)
            for key, value in img_state.items():
                if key != "hidden" and key in img_states:
                    img_states[key].append(value)

        # Stack Lists
        img_states = {k: torch.concat(v, dim=1) for k, v in img_states.items()}  # (B, 1+img_steps, D)

        return img_states

    def get_feat(self, state, blocks_deter_id=None):

        return torch.cat([state["stoch"].flatten(start_dim=-2, end_dim=-1), state["deter"] if blocks_deter_id is None else state["blocks_deter"][blocks_deter_id]], dim=-1)
    
    def get_dist(self, state):

        return torch.distributions.Independent(distributions.OneHotDist(logits=state['logits'], uniform_mix=self.uniform_mix), 1)

    def slice_hidden(self, hidden):

        hidden = [(hidden_blk[0][:, -self.att_context_left:], hidden_blk[1][:, -self.att_context_left:]) for hidden_blk in hidden]

        return hidden
    
    def get_hidden_len(self, hidden):

        if hidden != None:
            return hidden[0][0].shape[1]
        else:
            return 0

    def forward_img(self, prev_states, prev_actions, mask, return_att_w=False, return_blocks_deter=False):
        """
        Forward pass for imagination mode using motion features.
        Expects prev_states to contain both z_t and z_{t-1}.
        """
        # Clip Action -c:+c
        if self.action_clip > 0.0:
            prev_actions = prev_actions * (self.action_clip / torch.clip(torch.abs(prev_actions), min=self.action_clip)).detach()

        # Flatten stoch size and discrete size to get z_{t-1} and z_t
        if self.discrete:
            # prev_states["stoch"] shape: (B, 2, stoch_size, discrete)
            stoch_prev = prev_states["stoch"][:, 0].flatten(start_dim=-2, end_dim=-1)  # z_{t-1}
            stoch_current = prev_states["stoch"][:, 1].flatten(start_dim=-2, end_dim=-1)  # z_t
        else:
            # prev_states["stoch"] shape: (B, 2, stoch_size)
            stoch_prev = prev_states["stoch"][:, 0]  # z_{t-1}
            stoch_current = prev_states["stoch"][:, 1]  # z_t

        # Compute motion: z_t vs z_{t-1}
        motion = self.compute_motion(stoch_current, stoch_prev)

        # Mix motion with action: motion ⊕ action
        motion_action = self.motion_action_mixer(torch.concat([motion, prev_actions], dim=-1))

        # Transformer processes motion: motion_action -> m_t
        assert self.get_hidden_len(prev_states["hidden"]) <= self.att_context_left, \
            "warning: att context left is {} and hidden has length {}".format(
                self.att_context_left, self.get_hidden_len(prev_states["hidden"]))
        
        outputs = self.transformer(motion_action, hidden=prev_states["hidden"], mask=mask, 
                                return_hidden=True, return_att_w=return_att_w, 
                                return_blocks_x=return_blocks_deter)
        m_t, hidden = outputs.x, outputs.hidden

        # Additional Outputs
        add_out_dict = {}
        if return_att_w:
            add_out_dict["att_w"] = outputs.att_w
        if return_blocks_deter:
            add_out_dict["blocks_deter"] = outputs.blocks_x

        # Fuse m_t with z_{t-1}: [m_t ⊕ z_{t-1}]
        fused = self.dynamics_fusion(torch.concat([m_t, stoch_prev], dim=-1))

        # Predict z_t distribution
        logits = self.dynamics_predictor(fused).reshape(fused.shape[:-1] + (self.stoch_size, self.discrete))
        dist_params = {'logits': logits}

        # Sample
        stoch = self.get_dist(dist_params).rsample()

        # Return Prior (deter now stores the fused representation)
        return {"stoch": stoch, "deter": fused, "hidden": hidden, **dist_params, **add_out_dict}
    
    def forward_obs(self, deter, hidden, states):
        
        # Return Post
        return {"deter": deter, "hidden": hidden, **states}

    def forward(self, states, prev_states, prev_actions, is_firsts, is_firsts_hidden=None, 
            return_att_w=False, return_blocks_deter=False):
        """
        Forward pass that processes motion between consecutive states.
        """
        # (B, 1 or L, A)
        assert prev_actions.dim() == 3 
        # (B, 1 or L)
        assert is_firsts.dim() == 2

        # Clip Action (B, L, D) -c:+c
        if self.action_clip > 0.0:
            prev_actions *= (self.action_clip / torch.clip(torch.abs(prev_actions), min=self.action_clip)).detach()

        # Create right context mask (B, 1, L, Th+L)
        mask = modules.return_mask(seq_len=prev_actions.shape[1], 
                                hidden_len=self.get_hidden_len(prev_states["hidden"]), 
                                left_context=self.att_context_left, right_context=0, 
                                dtype=prev_actions.dtype, device=prev_actions.device)

        # 1: Reset First States and Actions
        # 2: Update mask to mask pre is_first positions
        if is_firsts.any():
            # Unsqueeze is_firsts (B, L, 1)
            is_firsts = is_firsts.unsqueeze(dim=-1)

            # Reset first Actions
            prev_actions *= (1.0 - is_firsts)

            # Reset first States (B, L, ...)
            init_state = self.initial(batch_size=prev_actions.shape[0], seq_length=prev_actions.shape[1], 
                                    dtype=prev_actions.dtype, device=prev_actions.device)
            for key, value in prev_states.items():
                # Hidden does not need reset, auto masked
                if key == "hidden":
                    prev_states[key] = value
                # Reset first States 
                else:
                    is_firsts_r = torch.reshape(is_firsts, is_firsts.shape + (1,) * (len(value.shape) - len(is_firsts.shape)))
                    prev_states[key] = value * (1.0 - is_firsts_r) + init_state[key] * is_firsts_r

            # Mask positions of past trajectories # (B, 1, L, Th+L)
            is_firts_mask = modules.return_is_firsts_mask(is_firsts.squeeze(dim=-1), is_firsts_hidden=is_firsts_hidden)
            mask = mask.minimum(is_firts_mask)

        # Flatten current and previous stoch for motion computation
        if self.discrete:
            stoch_current = states["stoch"].flatten(start_dim=-2, end_dim=-1)
            stoch_prev = prev_states["stoch"].flatten(start_dim=-2, end_dim=-1)
        else:
            stoch_current = states["stoch"]
            stoch_prev = prev_states["stoch"]

        # Compute motion between current and previous states
        motion = self.compute_motion(stoch_current, stoch_prev)

        # Mix motion with action: motion ⊕ action
        motion_action = self.motion_action_mixer(torch.concat([motion, prev_actions], dim=-1))

        # Transformer processes motion: motion_action -> m_t
        outputs = self.transformer(motion_action, hidden=prev_states["hidden"], mask=mask, 
                                return_hidden=True, return_att_w=return_att_w, 
                                return_blocks_x=return_blocks_deter)
        m_t, hidden = outputs.x, outputs.hidden

        # Additional Outputs
        add_out_dict = {}
        if return_att_w:
            add_out_dict["att_w"] = outputs.att_w
        if return_blocks_deter:
            add_out_dict["blocks_deter"] = outputs.blocks_x

        # Fuse m_t with z_{t-1}: [m_t ⊕ z_{t-1}]
        fused = self.dynamics_fusion(torch.concat([m_t, stoch_prev], dim=-1))

        # Predict prior z_t distribution
        logits_prior = self.dynamics_predictor(fused).reshape(fused.shape[:-1] + (self.stoch_size, self.discrete))
        
        # Create prior dict
        prior = {
            "stoch": self.get_dist({'logits': logits_prior}).rsample(),
            "deter": fused,
            "hidden": hidden,
            "logits": logits_prior,
            **add_out_dict
        }

        # Forward Obs - post uses same deter and hidden
        post = self.forward_obs(fused, hidden, states)
        if return_att_w:
            post["att_w"] = prior["att_w"]
        if return_blocks_deter:
            post["blocks_deter"] = prior["blocks_deter"]

        # Return post and prior
        return post, prior