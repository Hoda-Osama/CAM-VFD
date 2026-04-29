import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal_model import TemporalModel


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module with CMAD (Cross-Modal Attention Discrepancy) support"""

    def __init__(self, app_dim, motion_dim, depth_dim, hidden_dim=256, num_heads=8, dropout=0.5, cmad_enabled=True):
        super().__init__()
        self.cmad_enabled = cmad_enabled

        # Projection layers
        self.app_proj = nn.Linear(app_dim, hidden_dim)
        self.motion_proj = nn.Linear(motion_dim, hidden_dim)
        self.depth_proj = nn.Linear(depth_dim, hidden_dim)

        # Temporal modeling
        self.temporal_model = TemporalModel(input_dim=hidden_dim)

        # Cross-attention layers
        self.app_motion_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.app_depth_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # Layer normalization for CMAD computation
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Fusion classifier (only these are trainable when backbones are frozen)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, 1)
        )

        # Optional: Additional projection for CMAD features
        if cmad_enabled:
            self.cmad_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, app_feats, depth_feats, motion_feats, return_cmad=False):
        """
        Forward pass with optional CMAD computation

        Args:
            app_feats: Appearance features (B, T, app_dim)
            depth_feats: Depth features (B, depth_dim)
            motion_feats: Motion features (B, motion_dim)
            return_cmad: Whether to return CMAD values

        Returns:
            classification output, and optionally CMAD values and attention maps
        """
        # Project features
        app = self.app_proj(app_feats)  # (B, T, hidden_dim)
        motion = self.motion_proj(motion_feats)  # (B, hidden_dim)
        depth = self.depth_proj(depth_feats)  # (B, hidden_dim)

        # Temporal modeling of appearance
        app_temp = self.temporal_model(app)  # (B, T, hidden_dim)

        # Cross-attention: Appearance-Motion
        motion_expanded = motion.unsqueeze(1)  # (B, 1, hidden_dim)
        app_motion, motion_attention_weights = self.app_motion_attn(
            query=app_temp,
            key=motion_expanded,
            value=motion_expanded
        )

        # Cross-attention: Appearance-Depth
        depth_expanded = depth.unsqueeze(1)  # (B, 1, hidden_dim)
        app_depth, depth_attention_weights = self.app_depth_attn(
            query=app_temp,
            key=depth_expanded,
            value=depth_expanded
        )

        # Compute CMAD (Cross-Modal Attention Discrepancy) if enabled
        cmad_value = None
        if self.cmad_enabled and return_cmad:
            cmad_value = self.compute_cmad(app_temp, app_motion, app_depth)

        # Pool features for classification
        app_pool = app_temp.mean(dim=1)
        app_motion_pool = app_motion.mean(dim=1)
        app_depth_pool = app_depth.mean(dim=1)

        # Combine and classify
        combined = torch.cat([app_pool, app_motion_pool, app_depth_pool], dim=1)
        output = self.classifier(combined).squeeze(1)

        if return_cmad:
            return output, cmad_value, {
                'app_motion_attention': motion_attention_weights,
                'app_depth_attention': depth_attention_weights,
                'app_temp_features': app_temp,
                'app_motion_features': app_motion,
                'app_depth_features': app_depth
            }

        return output

    def compute_cmad(self, app_temp, app_motion, app_depth):
        """
        Compute Cross-Modal Attention Discrepancy (CMAD)

        CMAD measures cross-modal inconsistency by computing the discrepancy
        between appearance-motion and appearance-depth attention maps.

        Formula:
        CMAD(V) = (1/(2T)) * Σ_t [ ||H_app-motion^(t) - H_app^temp,(t)||^2 +
                                      ||H_app-depth^(t) - H_app^temp,(t)||^2 ]

        Args:
            app_temp: Temporal appearance features (B, T, hidden_dim)
            app_motion: Appearance-motion cross-attended features (B, T, hidden_dim)
            app_depth: Appearance-depth cross-attended features (B, T, hidden_dim)

        Returns:
            cmad: Cross-modal attention discrepancy value (scalar per video)
        """
        # Compute L2 discrepancies
        motion_discrepancy = torch.norm(app_motion - app_temp, p=2, dim=-1) ** 2
        depth_discrepancy = torch.norm(app_depth - app_temp, p=2, dim=-1) ** 2

        # Average over temporal dimension and combine
        cmad = (motion_discrepancy + depth_discrepancy).mean(dim=1) / 2

        # Normalize if requested
        if hasattr(self, 'layer_norm'):
            cmad = self.layer_norm(cmad.unsqueeze(-1)).squeeze(-1)

        return cmad  # (B,)

    def get_cmad_score(self, app_feats, depth_feats, motion_feats):
        """
        Get CMAD score without classification head
        Useful for analysis and validation
        """
        app = self.app_proj(app_feats)
        motion = self.motion_proj(motion_feats)
        depth = self.depth_proj(depth_feats)

        app_temp = self.temporal_model(app)

        motion_expanded = motion.unsqueeze(1)
        app_motion, _ = self.app_motion_attn(
            query=app_temp,
            key=motion_expanded,
            value=motion_expanded
        )

        depth_expanded = depth.unsqueeze(1)
        app_depth, _ = self.app_depth_attn(
            query=app_temp,
            key=depth_expanded,
            value=depth_expanded
        )

        return self.compute_cmad(app_temp, app_motion, app_depth)


class CMADAnalyzer(nn.Module):
    """
    Standalone CMAD analyzer for computing discrepancy scores
    without the classification head
    """

    def __init__(self, app_dim, motion_dim, depth_dim, hidden_dim=256, num_heads=8):
        super().__init__()
        self.app_proj = nn.Linear(app_dim, hidden_dim)
        self.motion_proj = nn.Linear(motion_dim, hidden_dim)
        self.depth_proj = nn.Linear(depth_dim, hidden_dim)

        self.temporal_model = TemporalModel(input_dim=hidden_dim)
        self.app_motion_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.app_depth_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, app_feats, depth_feats, motion_feats):
        app = self.app_proj(app_feats)
        motion = self.motion_proj(motion_feats)
        depth = self.depth_proj(depth_feats)

        app_temp = self.temporal_model(app)

        motion_expanded = motion.unsqueeze(1)
        app_motion, _ = self.app_motion_attn(
            query=app_temp,
            key=motion_expanded,
            value=motion_expanded
        )

        depth_expanded = depth.unsqueeze(1)
        app_depth, _ = self.app_depth_attn(
            query=app_temp,
            key=depth_expanded,
            value=depth_expanded
        )

        # Compute CMAD
        motion_discrepancy = torch.norm(app_motion - app_temp, p=2, dim=-1) ** 2
        depth_discrepancy = torch.norm(app_depth - app_temp, p=2, dim=-1) ** 2

        cmad = (motion_discrepancy + depth_discrepancy).mean(dim=1) / 2

        return cmad, app_temp, app_motion, app_depth