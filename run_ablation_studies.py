"""
Main script to run all ablation studies:
1. Modality Ablation
2. Temporal Ablation
3. Fusion Strategy and Query Direction Ablation
"""

import torch
import yaml
import argparse
import logging
from data.dataset import load_multi_datasets, create_dataloaders
from models import CLIPAppearanceModel, EnhancedDepthModel, VideoMAEMotionModel, CrossAttentionFusion
from utils.ablation import ModalityAblation, TemporalAblation, FusionAblation
from utils import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--ablation_type', type=str, default='all',
                        choices=['modality', 'temporal', 'fusion', 'all'],
                        help='Type of ablation study to run')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup
    set_seed(42)
    device = get_device()
    logging.basicConfig(level=logging.INFO)

    print(f"Using device: {device}")
    print(f"Ablation Study Type: {args.ablation_type}")

    # Load datasets
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, dataset_stats = load_multi_datasets(config, device)

    # Initialize feature extractors
    print("\nInitializing feature extractors...")
    appearance_model = CLIPAppearanceModel(device=device).to(device).eval()
    depth_model = EnhancedDepthModel().to(device).eval()
    motion_model = VideoMAEMotionModel(
        freeze_layers=config['model']['motion']['freeze_layers'],
        device=device
    ).to(device).eval()

    # Create base fusion model
    base_fusion_model = CrossAttentionFusion(
        app_dim=config['model']['appearance']['feature_dim'],
        motion_dim=config['model']['motion']['feature_dim'],
        depth_dim=config['model']['depth']['feature_dim'],
        hidden_dim=config['model']['fusion']['hidden_dim'],
        num_heads=config['model']['fusion']['num_heads'],
        dropout=config['model']['fusion']['dropout']
    )

    results_summary = {}

    # Run Modality Ablation
    if args.ablation_type in ['modality', 'all']:
        print("\n" + "=" * 80)
        print("RUNNING MODALITY ABLATION STUDIES")
        print("=" * 80)

        modality_ablation = ModalityAblation(
            base_fusion_model, appearance_model, depth_model, motion_model,
            device, config, CrossAttentionFusion
        )

        modality_results = modality_ablation.run_all_ablations(
            train_loader, val_loader, test_loader
        )
        modality_ablation.print_summary()
        results_summary['modality'] = modality_results

    # Run Temporal Ablation
    if args.ablation_type in ['temporal', 'all']:
        print("\n" + "=" * 80)
        print("RUNNING TEMPORAL ABLATION STUDIES")
        print("=" * 80)

        temporal_ablation = TemporalAblation(
            base_fusion_model, appearance_model, depth_model, motion_model,
            device, config, CrossAttentionFusion
        )

        temporal_results = temporal_ablation.run_all_ablations(
            train_loader, val_loader, test_loader
        )
        temporal_ablation.print_summary()
        results_summary['temporal'] = temporal_results

    # Run Fusion Ablation
    if args.ablation_type in ['fusion', 'all']:
        print("\n" + "=" * 80)
        print("RUNNING FUSION STRATEGY ABLATION STUDIES")
        print("=" * 80)

        fusion_ablation = FusionAblation(
            base_fusion_model, appearance_model, depth_model, motion_model,
            device, config
        )

        fusion_results = fusion_ablation.run_all_ablations(
            train_loader, val_loader, test_loader
        )
        fusion_ablation.print_summary()
        results_summary['fusion'] = fusion_results

    # Save complete results
    torch.save(results_summary, 'complete_ablation_results.pth')
    print("\nAll ablation results saved to 'complete_ablation_results.pth'")

    # Generate comprehensive report
    generate_ablation_report(results_summary)


def generate_ablation_report(results):
    """Generate comprehensive ablation study report"""

    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("COMPREHENSIVE ABLATION STUDY REPORT")
    report_lines.append("=" * 100)
    report_lines.append("")

    # Modality Ablation Results
    if 'modality' in results:
        report_lines.append("1. MODALITY ABLATION STUDY")
        report_lines.append("-" * 60)
        report_lines.append(f"{'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'Recall':<12} {'AUROC':<12}")
        report_lines.append("-" * 60)

        for config_name, result in results['modality'].items():
            metrics = result['test_metrics']
            report_lines.append(f"{result['name']:<25} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} "
                                f"{metrics['recall']:<12.4f} {metrics.get('auroc', 0):<12.4f}")
        report_lines.append("")

    # Temporal Ablation Results
    if 'temporal' in results:
        report_lines.append("2. TEMPORAL ABLATION STUDY")
        report_lines.append("-" * 60)
        report_lines.append(f"{'Frames':<12} {'Accuracy':<12} {'F1-Score':<12} {'Recall':<12} {'AUROC':<12}")
        report_lines.append("-" * 60)

        for frame_count, result in results['temporal'].items():
            metrics = result['test_metrics']
            report_lines.append(f"{frame_count:<12} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} "
                                f"{metrics['recall']:<12.4f} {metrics.get('auroc', 0):<12.4f}")
        report_lines.append("")

    # Fusion Ablation Results
    if 'fusion' in results:
        report_lines.append("3. FUSION STRATEGY ABLATION STUDY")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Fusion Strategy':<35} {'Accuracy':<12} {'F1-Score':<12} {'Recall':<12} {'AUROC':<12}")
        report_lines.append("-" * 80)

        for config_name, result in results['fusion'].items():
            metrics = result['test_metrics']
            report_lines.append(f"{result['name']:<35} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} "
                                f"{metrics['recall']:<12.4f} {metrics.get('auroc', 0):<12.4f}")
        report_lines.append("")

    # Summary and Recommendations
    report_lines.append("4. SUMMARY AND RECOMMENDATIONS")
    report_lines.append("-" * 60)

    # Find best models
    if 'modality' in results:
        best_modality = max(results['modality'].items(),
                            key=lambda x: x[1]['test_metrics']['accuracy'])
        report_lines.append(f"Best Modality Configuration: {best_modality[1]['name']} "
                            f"(Acc: {best_modality[1]['test_metrics']['accuracy']:.4f})")

    if 'temporal' in results:
        best_temporal = max(results['temporal'].items(),
                            key=lambda x: x[1]['test_metrics']['accuracy'])
        report_lines.append(f"Best Temporal Configuration: {best_temporal[0]} frames "
                            f"(Acc: {best_temporal[1]['test_metrics']['accuracy']:.4f})")

    if 'fusion' in results:
        best_fusion = max(results['fusion'].items(),
                          key=lambda x: x[1]['test_metrics']['accuracy'])
        report_lines.append(f"Best Fusion Strategy: {best_fusion[1]['name']} "
                            f"(Acc: {best_fusion[1]['test_metrics']['accuracy']:.4f})")

    report_lines.append("")
    report_lines.append("=" * 100)

    # Write report to file
    report_text = "\n".join(report_lines)
    with open('ablation_study_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)


if __name__ == "__main__":
    main()