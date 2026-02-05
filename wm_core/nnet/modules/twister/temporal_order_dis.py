# temporal_order_discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import math
import random

class TemporalOrderDiscriminator(nn.Module): 
    def __init__(self, feat_dim=128, window_size=4, hidden_dim=128, num_heads=8, num_layers=2, max_perms=1000): 
        super().__init__() 
 
        self.window_size = window_size 
        total_perms = math.factorial(window_size)
        
        # Limit number of permutations to max_perms
        self.num_perm = min(total_perms, max_perms)
        # print(f"self.num_perm: {self.num_perm}")
        self.selected_perms = set()

        while len(self.selected_perms) < self.num_perm:
            perm = tuple(random.sample(range(window_size), window_size))
            self.selected_perms.add(perm)  # set ensures uniqueness
        self.selected_perms = list(self.selected_perms)
        
        # Store as tensor for faster lookup
        self.register_buffer(
            'perm_indices', 
            torch.tensor(self.selected_perms, dtype=torch.long)
        )

        self.input_proj = nn.Linear(1024, feat_dim)
         
        encoder_layer = nn.TransformerEncoderLayer( 
            d_model=feat_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=0.1,
            activation="gelu", 
            batch_first=True 
        ) 
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) 
 
        # Classification head 
        self.head = nn.Sequential( 
            # nn.Linear(feat_dim, hidden_dim), 
            # nn.ReLU(), 
            nn.Linear(hidden_dim, self.num_perm) 
        ) 
 
    def forward(self, x): 
        # x: (B, k, feat_dim)
        x = self.input_proj(x) 
        h = self.encoder(x)           # (B, k, feat_dim) 
        h = h.mean(1)                 # pool 
        logits = self.head(h)         # (B, num_perm) 
        return logits
    
    def get_permutation(self, perm_id):
        """Get the permutation tuple for a given ID"""
        return tuple(self.perm_indices[perm_id].tolist())