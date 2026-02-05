import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class TemporalDiscriminator(nn.Module):
    
    def __init__(self,
                 dim_real=1024,
                 proj_dim=192,
                 hidden_dim=256,
                 num_layers=1,
                 num_heads=1,
                 dropout=0.05):
        super().__init__()

        # 1) Small projection
        self.proj = nn.Sequential(
            spectral_norm(nn.Linear(dim_real, proj_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2) Lightweight temporal conv
        self.temporal_conv = nn.Sequential(
            spectral_norm(nn.Conv1d(proj_dim, hidden_dim, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 3) Small transformer (only 1 layer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,  # 384 instead of 1024+
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) Final classifier
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),  # 192 → 96
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden_dim // 2, 1)),
        )

    def forward(self, z):
        """
        z: [B, L, D]
        """
        x = self.proj(z)              # (B, L, 128)

        x = x.transpose(1, 2)         # (B, 128, L)
        x = self.temporal_conv(x)     # (B, 192, L)
        x = x.transpose(1, 2)         # (B, L, 192)

        x = self.transformer(x)       # (B, L, 192)
        x = x.mean(dim=1)             # (B, 192)

        logits = self.fc(x)           # (B, 1)
        return logits.squeeze(-1)
