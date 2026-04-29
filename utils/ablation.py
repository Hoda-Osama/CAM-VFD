"""
Ablation Studies Module for Deepfake Detection
Implements various ablation experiments:
1. Modality Ablation
2. Temporal Ablation
3. Fusion Strategy and Query Direction Ablation
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score


class ModalityAblation:
    """Modality ablation experiments"""

    MODALITY_CONFIGS = {
        'appearance_only': {
            'use_appearance': True,
            'use_motion': False,
            'use_depth': False,
            'name': 'Appearance Only'
        },
        'motion_only': {
            'use_appearance': False,
            'use_motion': True,
            'use_depth': False,
            'name': 'Motion Only'
        },
        'depth_only': {
            'use_appearance': False,
            'use_motion': False,
            'use_depth': True,
            'name': 'Depth Only'
        },
        'appearance_motion': {
            'use_appearance': True,
            'use_motion': True,
            'use_depth': False,
            'name': 'Appearance + Motion'
        },
        'appearance_depth': {
            'use_appearance': True,
            'use_motion': False,
            'use_depth': True,
            'name': 'Appearance + Depth'
        },
        'motion_depth': {
            'use_appearance': False,
            'use_motion': True,
            'use_depth': True,
            'name': 'Motion + Depth'
        },
        'full_model': {
            'use_appearance': True,
            'use_motion': True,
            'use_depth': True,
            'name': 'Full Model (All Modalities)'
        }
    }

    def __init__(self, base_model, appearance_model, depth_model, motion_model,
                 device, config, fusion_class):
        """
        Initialize modality ablation experiments

        Args:
            base_model: Base fusion model template
            appearance_model: CLIP appearance model
            depth_model: Depth model
            motion_model: Motion model
            device: Device to run on
            config: Configuration dictionary
            fusion_class: Fusion model class
        """
        self.base_model = base_model
        self.appearance_model = appearance_model
        self.depth_model = depth_model
        self.motion_model = motion_model
        self.device = device
        self.config = config
        self.fusion_class = fusion_class
        self.results = {}

    def create_ablation_model(self, config_name):
        """Create model with specific modalities"""
        cfg = self.MODALITY_CONFIGS[config_name]

        # Get feature dimensions based on active modalities
        app_dim = self.config['model']['appearance']['feature_dim'] if cfg['use_appearance'] else 0
        motion_dim = self.config['model']['motion']['feature_dim'] if cfg['use_motion'] else 0
        depth_dim = self.config['model']['depth']['feature_dim'] if cfg['use_depth'] else 0

        # Create fusion model with appropriate dimensions
        model = AblationFusionModel(
            app_dim=app_dim,
            motion_dim=motion_dim,
            depth_dim=depth_dim,
            hidden_dim=self.config['model']['fusion']['hidden_dim'],
            num_heads=self.config['model']['fusion']['num_heads'],
            dropout=self.config['model']['fusion']['dropout'],
            use_appearance=cfg['use_appearance'],
            use_motion=cfg['use_motion'],
            use_depth=cfg['use_depth']
        ).to(self.device)

        return model, cfg

    def train_ablation_model(self, model, train_loader, val_loader, config_name, epochs=50):
        """Train an ablation model"""
        from .training import AblationTrainer

        print(f"\n{'=' * 60}")
        print(f"Training: {self.MODALITY_CONFIGS[config_name]['name']}")
        print(f"{'=' * 60}")

        trainer = AblationTrainer(
            model, self.appearance_model, self.depth_model, self.motion_model,
            train_loader, val_loader, self.config, self.device,
            use_appearance=self.MODALITY_CONFIGS[config_name]['use_appearance'],
            use_motion=self.MODALITY_CONFIGS[config_name]['use_motion'],
            use_depth=self.MODALITY_CONFIGS[config_name]['use_depth']
        )

        trained_model, metrics = trainer.train()
        return trained_model, metrics

    def run_all_ablations(self, train_loader, val_loader, test_loader):
        """Run all modality ablation experiments"""
        results = {}

        for config_name in self.MODALITY_CONFIGS.keys():
            print(f"\n{'=' * 60}")
            print(f"Starting Modality Ablation: {self.MODALITY_CONFIGS[config_name]['name']}")
            print(f"{'=' * 60}")

            # Create model
            model, cfg = self.create_ablation_model(config_name)

            # Train model
            trained_model, train_metrics = self.train_ablation_model(
                model, train_loader, val_loader, config_name
            )

            # Evaluate on test set
            test_metrics = evaluate_ablation_model(
                trained_model, self.appearance_model, self.depth_model, self.motion_model,
                test_loader, self.device,
                use_appearance=cfg['use_appearance'],
                use_motion=cfg['use_motion'],
                use_depth=cfg['use_depth']
            )

            results[config_name] = {
                'name': self.MODALITY_CONFIGS[config_name]['name'],
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'config': cfg
            }

            # Save results
            self.save_results(results)

        self.results = results
        return results

    def save_results(self, results, filename='ablation_results_modality.pth'):
        """Save ablation results"""
        torch.save(results, filename)
        print(f"\nResults saved to {filename}")

    def print_summary(self):
        """Print ablation study summary"""
        print("\n" + "=" * 80)
        print("MODALITY ABLATION STUDY SUMMARY")
        print("=" * 80)
        print(f"{'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'Recall':<12} {'AUROC':<12}")
        print("-" * 80)

        for config_name, result in self.results.items():
            metrics = result['test_metrics']
            print(f"{result['name']:<25} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics.get('auroc', 0):<12.4f}")

        print("=" * 80)


class TemporalAblation:
    """Temporal ablation experiments (8, 16, 32 frames)"""

    FRAME_COUNTS = [8, 16, 32]

    def __init__(self, base_model, appearance_model, depth_model, motion_model,
                 device, config, fusion_class):
        self.base_model = base_model
        self.appearance_model = appearance_model
        self.depth_model = depth_model
        self.motion_model = motion_model
        self.device = device
        self.config = config
        self.fusion_class = fusion_class
        self.results = {}

    def run_ablation(self, train_loader_template, val_loader_template, test_loader_template, frame_count):
        """Run ablation for specific frame count"""
        print(f"\n{'=' * 60}")
        print(f"Temporal Ablation: {frame_count} frames")
        print(f"{'=' * 60}")

        # Create model with full modalities
        model = self.fusion_class(
            app_dim=self.config['model']['appearance']['feature_dim'],
            motion_dim=self.config['model']['motion']['feature_dim'],
            depth_dim=self.config['model']['depth']['feature_dim'],
            hidden_dim=self.config['model']['fusion']['hidden_dim'],
            num_heads=self.config['model']['fusion']['num_heads'],
            dropout=self.config['model']['fusion']['dropout']
        ).to(self.device)

        # Modify dataloaders to use different frame count
        train_loader = self._modify_dataloader_frames(train_loader_template, frame_count)
        val_loader = self._modify_dataloader_frames(val_loader_template, frame_count)
        test_loader = self._modify_dataloader_frames(test_loader_template, frame_count)

        # Train model
        from .training import Trainer
        trainer = Trainer(
            model, self.appearance_model, self.depth_model, self.motion_model,
            train_loader, val_loader, self.config, self.device
        )

        trained_model, metrics = trainer.train()

        # Evaluate
        test_metrics = evaluate_model_standard(
            trained_model, self.appearance_model, self.depth_model, self.motion_model,
            test_loader, self.device
        )

        self.results[frame_count] = {
            'frame_count': frame_count,
            'train_metrics': metrics,
            'test_metrics': test_metrics
        }

        return test_metrics

    def _modify_dataloader_frames(self, dataloader, new_frame_count):
        """Modify dataloader to use different frame count"""
        # This requires recreating the dataset with new frame count
        # Implementation depends on your dataset structure
        return dataloader

    def run_all_ablations(self, train_loader, val_loader, test_loader):
        """Run all temporal ablations"""
        for frame_count in self.FRAME_COUNTS:
            self.run_ablation(train_loader, val_loader, test_loader, frame_count)
        self.save_results()
        return self.results

    def save_results(self, filename='ablation_results_temporal.pth'):
        torch.save(self.results, filename)

    def print_summary(self):
        print("\n" + "=" * 80)
        print("TEMPORAL ABLATION STUDY SUMMARY")
        print("=" * 80)
        print(f"{'Frames':<12} {'Accuracy':<12} {'F1-Score':<12} {'Recall':<12} {'AUROC':<12}")
        print("-" * 80)

        for frame_count, result in self.results.items():
            metrics = result['test_metrics']
            print(f"{frame_count:<12} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics.get('auroc', 0):<12.4f}")

        print("=" * 80)


class FusionAblation:
    """Fusion strategy and query direction ablation"""

    FUSION_CONFIGS = {
        'concatenation': {
            'type': 'concatenation',
            'name': 'Concatenation (Early Fusion)',
            'description': 'Simple concatenation of all features'
        },
        'late_fusion': {
            'type': 'late_fusion',
            'name': 'Late Fusion (Score Averaging)',
            'description': 'Average predictions from modality-specific classifiers'
        },
        'symmetric_cross_attention': {
            'type': 'symmetric',
            'name': 'Symmetric Cross-Attention',
            'description': 'Bidirectional cross-attention between all modalities'
        },
        'motion_queries_appearance_depth': {
            'type': 'motion_queries',
            'query_modality': 'motion',
            'key_value_modalities': ['appearance', 'depth'],
            'name': 'Motion Queries (Appearance, Depth)',
            'description': 'Motion as query, appearance and depth as key/value'
        },
        'appearance_queries_motion_depth': {
            'type': 'appearance_queries',
            'query_modality': 'appearance',
            'key_value_modalities': ['motion', 'depth'],
            'name': 'Appearance Queries (Motion, Depth)',
            'description': 'Appearance as query, motion and depth as key/value'
        },
        'depth_queries_appearance_motion': {
            'type': 'depth_queries',
            'query_modality': 'depth',
            'key_value_modalities': ['appearance', 'motion'],
            'name': 'Depth Queries (Appearance, Motion)',
            'description': 'Depth as query, appearance and motion as key/value'
        }
    }

    def __init__(self, base_model, appearance_model, depth_model, motion_model,
                 device, config):
        self.base_model = base_model
        self.appearance_model = appearance_model
        self.depth_model = depth_model
        self.motion_model = motion_model
        self.device = device
        self.config = config
        self.results = {}

    def create_fusion_model(self, config_name):
        """Create model with specific fusion strategy"""
        cfg = self.FUSION_CONFIGS[config_name]

        if cfg['type'] == 'concatenation':
            model = ConcatenationFusion(
                app_dim=self.config['model']['appearance']['feature_dim'],
                motion_dim=self.config['model']['motion']['feature_dim'],
                depth_dim=self.config['model']['depth']['feature_dim'],
                hidden_dim=self.config['model']['fusion']['hidden_dim'],
                dropout=self.config['model']['fusion']['dropout']
            ).to(self.device)

        elif cfg['type'] == 'late_fusion':
            model = LateFusionModel(
                app_dim=self.config['model']['appearance']['feature_dim'],
                motion_dim=self.config['model']['motion']['feature_dim'],
                depth_dim=self.config['model']['depth']['feature_dim'],
                hidden_dim=self.config['model']['fusion']['hidden_dim'],
                dropout=self.config['model']['fusion']['dropout']
            ).to(self.device)

        elif cfg['type'] == 'symmetric':
            model = SymmetricCrossAttentionFusion(
                app_dim=self.config['model']['appearance']['feature_dim'],
                motion_dim=self.config['model']['motion']['feature_dim'],
                depth_dim=self.config['model']['depth']['feature_dim'],
                hidden_dim=self.config['model']['fusion']['hidden_dim'],
                num_heads=self.config['model']['fusion']['num_heads'],
                dropout=self.config['model']['fusion']['dropout']
            ).to(self.device)

        else:  # Query-based fusion
            model = QueryBasedFusion(
                query_modality=cfg['query_modality'],
                key_value_modalities=cfg['key_value_modalities'],
                app_dim=self.config['model']['appearance']['feature_dim'],
                motion_dim=self.config['model']['motion']['feature_dim'],
                depth_dim=self.config['model']['depth']['feature_dim'],
                hidden_dim=self.config['model']['fusion']['hidden_dim'],
                num_heads=self.config['model']['fusion']['num_heads'],
                dropout=self.config['model']['fusion']['dropout']
            ).to(self.device)

        return model, cfg

    def run_ablation(self, train_loader, val_loader, test_loader, config_name):
        """Run ablation for specific fusion strategy"""
        print(f"\n{'=' * 60}")
        print(f"Fusion Ablation: {self.FUSION_CONFIGS[config_name]['name']}")
        print(f"{'=' * 60}")

        model, cfg = self.create_fusion_model(config_name)

        from .training import Trainer
        trainer = Trainer(
            model, self.appearance_model, self.depth_model, self.motion_model,
            train_loader, val_loader, self.config, self.device
        )

        trained_model, metrics = trainer.train()

        test_metrics = evaluate_model_standard(
            trained_model, self.appearance_model, self.depth_model, self.motion_model,
            test_loader, self.device
        )

        self.results[config_name] = {
            'name': cfg['name'],
            'train_metrics': metrics,
            'test_metrics': test_metrics,
            'config': cfg
        }

        return test_metrics

    def run_all_ablations(self, train_loader, val_loader, test_loader):
        """Run all fusion ablations"""
        for config_name in self.FUSION_CONFIGS.keys():
            self.run_ablation(train_loader, val_loader, test_loader, config_name)
        self.save_results()
        return self.results

    def save_results(self, filename='ablation_results_fusion.pth'):
        torch.save(self.results, filename)

    def print_summary(self):
        print("\n" + "=" * 100)
        print("FUSION STRATEGY ABLATION STUDY SUMMARY")
        print("=" * 100)
        print(f"{'Fusion Strategy':<35} {'Accuracy':<12} {'F1-Score':<12} {'Recall':<12} {'AUROC':<12}")
        print("-" * 100)

        for config_name, result in self.results.items():
            metrics = result['test_metrics']
            print(f"{result['name']:<35} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics.get('auroc', 0):<12.4f}")

        print("=" * 100)


# Helper Models for Ablation Studies

class AblationFusionModel(nn.Module):
    """Flexible fusion model for modality ablation"""

    def __init__(self, app_dim, motion_dim, depth_dim, hidden_dim=256,
                 num_heads=8, dropout=0.5, use_appearance=True,
                 use_motion=True, use_depth=True):
        super().__init__()

        self.use_appearance = use_appearance
        self.use_motion = use_motion
        self.use_depth = use_depth

        # Projection layers for active modalities
        if use_appearance:
            self.app_proj = nn.Linear(app_dim, hidden_dim)
        if use_motion:
            self.motion_proj = nn.Linear(motion_dim, hidden_dim)
        if use_depth:
            self.depth_proj = nn.Linear(depth_dim, hidden_dim)

        # Count active modalities
        self.num_modalities = sum([use_appearance, use_motion, use_depth])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, 1)
        )

    def forward(self, app_feats, depth_feats, motion_feats):
        features = []

        if self.use_appearance:
            # Temporal pooling for appearance features
            app = self.app_proj(app_feats)
            app_pooled = app.mean(dim=1)
            features.append(app_pooled)

        if self.use_motion:
            motion = self.motion_proj(motion_feats)
            features.append(motion)

        if self.use_depth:
            depth = self.depth_proj(depth_feats)
            features.append(depth)

        combined = torch.cat(features, dim=1)
        return self.classifier(combined).squeeze(1)


class ConcatenationFusion(nn.Module):
    """Early fusion via concatenation"""

    def __init__(self, app_dim, motion_dim, depth_dim, hidden_dim=256, dropout=0.5):
        super().__init__()

        total_dim = app_dim + motion_dim + depth_dim

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, 1)
        )

    def forward(self, app_feats, depth_feats, motion_feats):
        # Temporal pooling for appearance
        app_pooled = app_feats.mean(dim=1)

        # Concatenate all features
        combined = torch.cat([app_pooled, motion_feats, depth_feats], dim=1)
        return self.classifier(combined).squeeze(1)


class LateFusionModel(nn.Module):
    """Late fusion via score averaging"""

    def __init__(self, app_dim, motion_dim, depth_dim, hidden_dim=256, dropout=0.5):
        super().__init__()

        # Individual classifiers for each modality
        self.app_classifier = nn.Sequential(
            nn.Linear(app_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.motion_classifier = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.depth_classifier = nn.Sequential(
            nn.Linear(depth_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, app_feats, depth_feats, motion_feats):
        # Temporal pooling for appearance
        app_pooled = app_feats.mean(dim=1)

        # Get individual predictions
        app_out = self.app_classifier(app_pooled)
        motion_out = self.motion_classifier(motion_feats)
        depth_out = self.depth_classifier(depth_feats)

        # Average scores
        combined = (app_out + motion_out + depth_out) / 3
        return combined.squeeze(1)


class SymmetricCrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between all modalities"""

    def __init__(self, app_dim, motion_dim, depth_dim, hidden_dim=256, num_heads=8, dropout=0.5):
        super().__init__()

        self.app_proj = nn.Linear(app_dim, hidden_dim)
        self.motion_proj = nn.Linear(motion_dim, hidden_dim)
        self.depth_proj = nn.Linear(depth_dim, hidden_dim)

        # Cross-attention layers for all pairs
        self.app_motion_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.motion_app_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.app_depth_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.depth_app_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.motion_depth_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.depth_motion_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

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

    def forward(self, app_feats, depth_feats, motion_feats):
        app = self.app_proj(app_feats)
        motion = self.motion_proj(motion_feats).unsqueeze(1)
        depth = self.depth_proj(depth_feats).unsqueeze(1)

        # Symmetric cross-attention
        app_motion, _ = self.app_motion_attn(query=app, key=motion, value=motion)
        motion_app, _ = self.motion_app_attn(query=motion, key=app, value=app)

        app_depth, _ = self.app_depth_attn(query=app, key=depth, value=depth)
        depth_app, _ = self.depth_app_attn(query=depth, key=app, value=app)

        motion_depth, _ = self.motion_depth_attn(query=motion, key=depth, value=depth)
        depth_motion, _ = self.depth_motion_attn(query=depth, key=motion, value=motion)

        # Pool and combine
        app_pool = (app.mean(dim=1) + app_motion.mean(dim=1) + app_depth.mean(dim=1)) / 3
        motion_pool = (motion.squeeze(1) + motion_app.squeeze(1) + motion_depth.squeeze(1)) / 3
        depth_pool = (depth.squeeze(1) + depth_app.squeeze(1) + depth_motion.squeeze(1)) / 3

        combined = torch.cat([app_pool, motion_pool, depth_pool], dim=1)
        return self.classifier(combined).squeeze(1)


class QueryBasedFusion(nn.Module):
    """Flexible query-based cross-attention fusion"""

    def __init__(self, query_modality, key_value_modalities, app_dim, motion_dim,
                 depth_dim, hidden_dim=256, num_heads=8, dropout=0.5):
        super().__init__()

        self.query_modality = query_modality
        self.key_value_modalities = key_value_modalities

        # Projection layers
        self.app_proj = nn.Linear(app_dim, hidden_dim)
        self.motion_proj = nn.Linear(motion_dim, hidden_dim)
        self.depth_proj = nn.Linear(depth_dim, hidden_dim)

        # Cross-attention layers
        self.cross_attentions = nn.ModuleDict()
        for kv_modality in key_value_modalities:
            attn_name = f"{query_modality}_to_{kv_modality}"
            self.cross_attentions[attn_name] = nn.MultiheadAttention(
                hidden_dim, num_heads, batch_first=True
            )

        # Classifier
        num_features = hidden_dim * (1 + len(key_value_modalities))
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, 1)
        )

    def get_modality_features(self, app_feats, depth_feats, motion_feats):
        """Get projected features for each modality"""
        features = {}

        if self.query_modality == 'appearance' or 'appearance' in self.key_value_modalities:
            features['appearance'] = self.app_proj(app_feats)

        if self.query_modality == 'motion' or 'motion' in self.key_value_modalities:
            features['motion'] = self.motion_proj(motion_feats)
            features['motion_expanded'] = features['motion'].unsqueeze(1)

        if self.query_modality == 'depth' or 'depth' in self.key_value_modalities:
            features['depth'] = self.depth_proj(depth_feats)
            features['depth_expanded'] = features['depth'].unsqueeze(1)

        return features

    def forward(self, app_feats, depth_feats, motion_feats):
        features = self.get_modality_features(app_feats, depth_feats, motion_feats)

        # Get query features (with temporal dimension)
        if self.query_modality == 'appearance':
            query = features['appearance']  # (B, T, H)
        elif self.query_modality == 'motion':
            query = features['motion_expanded']  # (B, 1, H)
        elif self.query_modality == 'depth':
            query = features['depth_expanded']  # (B, 1, H)
        else:
            raise ValueError(f"Unknown query modality: {self.query_modality}")

        # Apply cross-attention with each key/value modality
        attended_features = [query.mean(dim=1)]  # Pooled query

        for kv_modality in self.key_value_modalities:
            # Get key/value features
            if kv_modality == 'appearance':
                kv = features['appearance']  # (B, T, H)
            elif kv_modality == 'motion':
                kv = features['motion_expanded']  # (B, 1, H)
            elif kv_modality == 'depth':
                kv = features['depth_expanded']  # (B, 1, H)

            # Apply cross-attention
            attn_name = f"{self.query_modality}_to_{kv_modality}"
            attended, _ = self.cross_attentions[attn_name](query=query, key=kv, value=kv)
            attended_features.append(attended.mean(dim=1))

        # Combine features
        combined = torch.cat(attended_features, dim=1)
        return self.classifier(combined).squeeze(1)


# Evaluation utilities

def evaluate_ablation_model(model, appearance_model, depth_model, motion_model,
                            loader, device, use_appearance=True, use_motion=True,
                            use_depth=True):
    """Evaluate ablation model"""
    model.eval()
    appearance_model.eval()
    depth_model.eval()
    motion_model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for frames, labels, _ in loader:
            frames = frames.to(device)

            # Extract features only for used modalities
            app_feats = appearance_model(frames) if use_appearance else None
            depth_feats = depth_model(frames) if use_depth else None
            motion_feats = motion_model(frames) if use_motion else None

            # Handle None values
            if app_feats is None:
                app_feats = torch.zeros(frames.shape[0], 16, 512).to(device)
            if depth_feats is None:
                depth_feats = torch.zeros(frames.shape[0], 128).to(device)
            if motion_feats is None:
                motion_feats = torch.zeros(frames.shape[0], 768).to(device)

            outputs = model(app_feats, depth_feats, motion_feats)
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    predictions = (np.array(all_probs) > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(all_labels, predictions),
        'f1': f1_score(all_labels, predictions, zero_division=0),
        'recall': recall_score(all_labels, predictions, zero_division=0),
        'precision': precision_score(all_labels, predictions, zero_division=0),
    }

    try:
        metrics['auroc'] = roc_auc_score(all_labels, all_probs)
    except:
        metrics['auroc'] = 0.0

    return metrics


def evaluate_model_standard(model, appearance_model, depth_model, motion_model, loader, device):
    """Standard evaluation for full model"""
    model.eval()
    appearance_model.eval()
    depth_model.eval()
    motion_model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for frames, labels, _ in loader:
            frames = frames.to(device)

            app_feats = appearance_model(frames)
            depth_feats = depth_model(frames)
            motion_feats = motion_model(frames)

            outputs = model(app_feats, depth_feats, motion_feats)
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    predictions = (np.array(all_probs) > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(all_labels, predictions),
        'f1': f1_score(all_labels, predictions, zero_division=0),
        'recall': recall_score(all_labels, predictions, zero_division=0),
        'precision': precision_score(all_labels, predictions, zero_division=0),
    }

    try:
        metrics['auroc'] = roc_auc_score(all_labels, all_probs)
    except:
        metrics['auroc'] = 0.0

    return metrics


class AblationTrainer:
    """Simplified trainer for ablation studies"""

    def __init__(self, model, appearance_model, depth_model, motion_model,
                 train_loader, val_loader, config, device,
                 use_appearance=True, use_motion=True, use_depth=True):
        self.model = model
        self.appearance_model = appearance_model
        self.depth_model = depth_model
        self.motion_model = motion_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_appearance = use_appearance
        self.use_motion = use_motion
        self.use_depth = use_depth

        # Freeze backbones
        for m in [appearance_model, depth_model, motion_model]:
            for param in m.parameters():
                param.requires_grad = False

        # Optimizer for trainable parameters
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['training']['epochs']
        )

        self.best_val_acc = 0
        self.metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def train(self):
        """Train the ablation model"""
        epochs = self.config['training']['epochs']
        patience = self.config['training']['early_stopping']['patience']
        no_improve = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_preds, train_labels = [], []

            for frames, labels, _ in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                frames = frames.to(self.device)
                labels = labels.to(self.device).float()

                with torch.no_grad():
                    app_feats = self.appearance_model(frames) if self.use_appearance else None
                    depth_feats = self.depth_model(frames) if self.use_depth else None
                    motion_feats = self.motion_model(frames) if self.use_motion else None

                    # Handle None values
                    if app_feats is None:
                        app_feats = torch.zeros(frames.shape[0], 16, 512).to(self.device)
                    if depth_feats is None:
                        depth_feats = torch.zeros(frames.shape[0], 128).to(self.device)
                    if motion_feats is None:
                        motion_feats = torch.zeros(frames.shape[0], 768).to(self.device)

                outputs = self.model(app_feats, depth_feats, motion_feats)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            # Validation
            val_loss, val_acc, val_f1 = self._validate()

            # Update metrics
            train_acc = accuracy_score(train_labels, train_preds)
            self.metrics['train_loss'].append(train_loss / len(self.train_loader))
            self.metrics['val_loss'].append(val_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_acc'].append(val_acc)

            # Learning rate scheduling
            self.scheduler.step()

            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                no_improve = 0
                torch.save(self.model.state_dict(), f"best_ablation_model.pth")
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            print(f"Epoch {epoch + 1}: Train Loss: {train_loss / len(self.train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Load best model
        self.model.load_state_dict(torch.load("best_ablation_model.pth"))
        return self.model, self.metrics

    def _validate(self):
        """Validation step"""
        self.model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for frames, labels, _ in self.val_loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device).float()

                app_feats = self.appearance_model(frames) if self.use_appearance else None
                depth_feats = self.depth_model(frames) if self.use_depth else None
                motion_feats = self.motion_model(frames) if self.use_motion else None

                if app_feats is None:
                    app_feats = torch.zeros(frames.shape[0], 16, 512).to(self.device)
                if depth_feats is None:
                    depth_feats = torch.zeros(frames.shape[0], 128).to(self.device)
                if motion_feats is None:
                    motion_feats = torch.zeros(frames.shape[0], 768).to(self.device)

                outputs = self.model(app_feats, depth_feats, motion_feats)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        return val_loss / len(self.val_loader), val_acc, val_f1