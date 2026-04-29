import torch
import torch.nn as nn


class TemporalModel(nn.Module):
    """Temporal modeling using Transformer encoder"""

    def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_heads=8):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_layers
        )
        self.positional_embed = nn.Parameter(torch.randn(1, 16, input_dim))

    def forward(self, x):
        """Apply temporal modeling to input features"""
        seq_len = x.size(1)
        pos_emb = self.positional_embed[:, :seq_len]
        x = x + pos_emb
        return self.transformer(x)