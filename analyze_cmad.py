"""
Script to analyze CMAD (Cross-Modal Attention Discrepancy) on GenVideo and GenVidBench datasets
"""

import torch
import yaml
import numpy as np
import pandas as pd
from data.dataset import load_multi_datasets, VideoDataset
from models import CLIPAppearanceModel, EnhancedDepthModel, VideoMAEMotionModel, CrossAttentionFusion, CMADAnalyzer
from utils import get_device


def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    # Initialize feature extractors
    print("Initializing feature extractors...")
    appearance_model = CLIPAppearanceModel(device=device).to(device).eval()
    depth_model = EnhancedDepthModel().to(device).eval()
    motion_model = VideoMAEMotionModel(
        freeze_layers=config['model']['motion']['freeze_layers'],
        device=device
    ).to(device).eval()

    # Initialize fusion model with CMAD
    fusion_model = CrossAttentionFusion(
        app_dim=config['model']['appearance']['feature_dim'],
        motion_dim=config['model']['motion']['feature_dim'],
        depth_dim=config['model']['depth']['feature_dim'],
        hidden_dim=config['model']['fusion']['hidden_dim'],
        num_heads=config['model']['fusion']['num_heads'],
        dropout=config['model']['fusion']['dropout'],
        cmad_enabled=True
    ).to(device)

    # Load trained model if exists
    try:
        fusion_model.load_state_dict(torch.load("best_model.pth", map_location=device))
        print("Loaded trained model")
    except:
        print("No trained model found. Using initialized model for CMAD analysis")

    # Initialize CMAD analyzer
    cmad_analyzer = CMADAnalyzer(
        fusion_model, appearance_model, depth_model, motion_model, device
    )

    # Analyze each dataset separately
    results = {}

    for dataset_name, dataset_config in config['data']['datasets'].items():
        if not dataset_config.get('enabled', True):
            continue

        print(f"\n{'=' * 60}")
        print(f"Analyzing {dataset_name}")
        print(f"{'=' * 60}")

        # Load test split for this dataset
        fake_videos = get_video_files(dataset_config['fake_path'])
        real_videos = get_video_files(dataset_config['real_path'])

        # Balance and split
        min_count = min(len(fake_videos), len(real_videos))
        fake_videos = random.sample(fake_videos, min_count)
        real_videos = random.sample(real_videos, min_count)

        all_videos = real_videos + fake_videos
        all_labels = [1] * len(real_videos) + [0] * len(fake_videos)

        # Use test ratio from config
        from sklearn.model_selection import train_test_split
        _, test_paths, _, test_labels = train_test_split(
            all_videos, all_labels, test_size=config['split']['test_split'],
            stratify=all_labels, random_state=config['split']['random_state']
        )

        test_dataset = VideoDataset(
            [p for p, l in zip(test_paths, test_labels) if l == 1],
            [p for p, l in zip(test_paths, test_labels) if l == 0],
            get_transforms('val'),
            frame_count=config['data']['frame_count'],
            mode='test',
            dataset_name=dataset_name
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers']
        )

        # Analyze CMAD
        stats, cmad_scores, labels = cmad_analyzer.analyze_dataset(test_loader, dataset_name)
        results[dataset_name] = {
            'stats': stats,
            'cmad_scores': cmad_scores,
            'labels': labels
        }

        # Save results
        np.savez(f'cmad_results_{dataset_name}.npz',
                 cmad_scores=cmad_scores,
                 labels=labels,
                 stats=stats)

    # Compare across datasets
    print(f"\n{'=' * 60}")
    print("Cross-Dataset CMAD Comparison")
    print(f"{'=' * 60}")

    comparison_data = []
    for dataset_name, result in results.items():
        stats = result['stats']
        comparison_data.append({
            'Dataset': dataset_name,
            'Real CMAD Mean': stats['real_mean'],
            'Real CMAD Std': stats['real_std'],
            'Fake CMAD Mean': stats['fake_mean'],
            'Fake CMAD Std': stats['fake_std'],
            'Cohen\'s d': stats['cohens_d'],
            'P-value': stats['p_value'],
            'ROC AUC': stats.get('roc_auc', 0)
        })

    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    df.to_csv('cmad_cross_dataset_comparison.csv', index=False)


if __name__ == "__main__":
    import random
    from torch.utils.data import DataLoader
    from data.transforms import get_transforms
    from data.dataset import get_video_files

    main()