"""
Main script for robustness evaluation:
1. Signal Degradation (Compression, Noise, Blur)
2. Photometric Perturbations (Lighting, Color Distortion)
3. Adversarial Attacks (FGSM, PGD-20)
"""

import torch
import yaml
import argparse
import logging
from data.dataset import load_multi_datasets, VideoDataset
from models import CLIPAppearanceModel, EnhancedDepthModel, VideoMAEMotionModel, CrossAttentionFusion
from utils.robustness import RobustnessEvaluator
from utils import set_seed, get_device
from torch.utils.data import DataLoader
from data.transforms import get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model robustness')
    parser.add_argument('--subset', type=str, default='both',
                        choices=['cogvideo', 'svd', 'both'],
                        help='Dataset subset to evaluate')
    parser.add_argument('--evaluation_type', type=str, default='all',
                        choices=['signal', 'photometric', 'adversarial', 'all'],
                        help='Type of robustness evaluation')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to trained model')
    return parser.parse_args()


def load_subset_dataset(config, subset_name, device):
    """Load specific subset (CogVideo or SVD) for evaluation"""

    # Get paths from config
    fake_dir = config['paths']['fake_videos']
    real_dir = config['paths']['real_videos']

    # Load all videos
    from data.dataset import get_video_files, load_dataset
    fake, real = load_dataset(fake_dir, real_dir)

    # Filter for specific subset if needed
    if subset_name == 'cogvideo':
        fake = [f for f in fake if 'cogvideo' in f.lower()]
        real = [r for r in real if 'cogvideo' in r.lower()]
    elif subset_name == 'svd':
        fake = [f for f in fake if 'svd' in f.lower()]
        real = [r for r in real if 'svd' in r.lower()]

    # Balance dataset
    min_count = min(len(fake), len(real))
    fake = fake[:min_count]
    real = real[:min_count]

    print(f"Loaded {subset_name.upper()} subset: {len(real)} real, {len(fake)} fake")

    # Create test dataset
    test_dataset = VideoDataset(
        real, fake,
        get_transforms('val'),
        frame_count=config['data']['frame_count'],
        mode='test',
        min_frames=config['data']['min_frames'],
        n_factor=config['data']['n_factor'],
        dataset_name=subset_name
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    return test_loader


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
    print(f"Evaluation Type: {args.evaluation_type}")
    print(f"Model Path: {args.model_path}")

    # Initialize feature extractors
    print("\nInitializing feature extractors...")
    appearance_model = CLIPAppearanceModel(device=device).to(device).eval()
    depth_model = EnhancedDepthModel().to(device).eval()
    motion_model = VideoMAEMotionModel(
        freeze_layers=config['model']['motion']['freeze_layers'],
        device=device
    ).to(device).eval()

    # Initialize fusion model
    fusion_model = CrossAttentionFusion(
        app_dim=config['model']['appearance']['feature_dim'],
        motion_dim=config['model']['motion']['feature_dim'],
        depth_dim=config['model']['depth']['feature_dim'],
        hidden_dim=config['model']['fusion']['hidden_dim'],
        num_heads=config['model']['fusion']['num_heads'],
        dropout=config['model']['fusion']['dropout'],
        cmad_enabled=config['model']['cmad']['enabled']
    ).to(device)

    # Load trained model
    print(f"Loading model from {args.model_path}...")
    state_dict = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        fusion_model.load_state_dict(state_dict['model_state_dict'])
    else:
        fusion_model.load_state_dict(state_dict)
    fusion_model.eval()

    # Initialize robustness evaluator
    evaluator = RobustnessEvaluator(
        fusion_model, appearance_model, depth_model, motion_model, device
    )

    # Determine subsets to evaluate
    subsets = []
    if args.subset == 'both':
        subsets = ['cogvideo', 'svd']
    else:
        subsets = [args.subset]

    # Run evaluations
    all_results = {}

    for subset_name in subsets:
        print("\n" + "=" * 80)
        print(f"EVALUATING ON {subset_name.upper()} SUBSET")
        print("=" * 80)

        # Load subset data
        test_loader = load_subset_dataset(config, subset_name, device)

        # Run requested evaluations
        results = {}

        if args.evaluation_type in ['signal', 'all']:
            print("\n--- Signal Degradation Evaluation ---")
            results['compression'] = evaluator.evaluate_compression(test_loader)
            results['noise'] = evaluator.evaluate_noise(test_loader)
            results['blur'] = evaluator.evaluate_blur(test_loader)

        if args.evaluation_type in ['photometric', 'all']:
            print("\n--- Photometric Perturbations Evaluation ---")
            results['photometric'] = evaluator.evaluate_photometric(test_loader)

        if args.evaluation_type in ['adversarial', 'all']:
            print("\n--- Adversarial Robustness Evaluation ---")
            results['adversarial'] = evaluator.evaluate_adversarial(test_loader, attack_type='both')

        # Generate full report
        if args.evaluation_type == 'all':
            evaluator.generate_full_report(test_loader, subset_name, 'robustness_report')

        all_results[subset_name] = results

    # Save results
    torch.save(all_results, f'robustness_results_{args.evaluation_type}.pth')
    print(f"\nResults saved to robustness_results_{args.evaluation_type}.pth")

    # Print summary tables
    print_summary_tables(all_results)


def print_summary_tables(results):
    """Print results in table format similar to the paper"""

    for subset_name, subset_results in results.items():
        print("\n" + "=" * 100)
        print(f"TABLE IX: SIGNAL DEGRADATION ROBUSTNESS - {subset_name.upper()}")
        print("=" * 100)

        # Compression
        if 'compression' in subset_results:
            print("\n--- H.264 Compression (Accuracy %) ---")
            print(f"{'CRF':<10}", end="")
            for crf in [18, 28, 32]:
                print(f"{crf:<12}", end="")
            print()
            print("-" * 40)
            print(f"{'CAM-VFD':<10}", end="")
            for crf in [18, 28, 32]:
                acc = subset_results['compression'][crf]['accuracy'] * 100
                print(f"{acc:<12.1f}", end="")
            print()

        # Noise
        if 'noise' in subset_results:
            print("\n--- Noise Robustness (Accuracy %) ---")
            print(f"{'Noise Type':<20} {'σ=0.03':<12} {'σ=0.1':<12} {'p=0.01':<12} {'p=0.05':<12}")
            print("-" * 70)

            gaussian_03 = subset_results['noise'].get('gaussian_noise_sigma_0.03', {}).get('accuracy', 0) * 100
            gaussian_01 = subset_results['noise'].get('gaussian_noise_sigma_0.1', {}).get('accuracy', 0) * 100
            sp_001 = subset_results['noise'].get('salt_pepper_noise_prob_0.01', {}).get('accuracy', 0) * 100
            sp_005 = subset_results['noise'].get('salt_pepper_noise_prob_0.05', {}).get('accuracy', 0) * 100

            print(f"{'Gaussian Noise':<20} {gaussian_03:<12.1f} {gaussian_01:<12.1f} {'-':<12} {'-':<12}")
            print(f"{'Salt-Pepper Noise':<20} {'-':<12} {'-':<12} {sp_001:<12.1f} {sp_005:<12.1f}")

        # Blur
        if 'blur' in subset_results:
            print("\n--- Blur Robustness (Accuracy %) ---")
            print(f"{'Blur Type':<20} {'σ=1':<12} {'σ=2':<12} {'r=3':<12} {'r=7':<12} {'L=7':<12} {'L=21':<12}")
            print("-" * 90)

            gb_1 = subset_results['blur'].get('gaussian_blur_sigma_1', {}).get('accuracy', 0) * 100
            gb_2 = subset_results['blur'].get('gaussian_blur_sigma_2', {}).get('accuracy', 0) * 100
            db_3 = subset_results['blur'].get('defocus_blur_radius_3', {}).get('accuracy', 0) * 100
            db_7 = subset_results['blur'].get('defocus_blur_radius_7', {}).get('accuracy', 0) * 100
            mb_7 = subset_results['blur'].get('motion_blur_length_7', {}).get('accuracy', 0) * 100
            mb_21 = subset_results['blur'].get('motion_blur_length_21', {}).get('accuracy', 0) * 100

            print(f"{'Gaussian Blur':<20} {gb_1:<12.1f} {gb_2:<12.1f} {'-':<12} {'-':<12} {'-':<12} {'-':<12}")
            print(f"{'Defocus Blur':<20} {'-':<12} {'-':<12} {db_3:<12.1f} {db_7:<12.1f} {'-':<12} {'-':<12}")
            print(f"{'Motion Blur':<20} {'-':<12} {'-':<12} {'-':<12} {'-':<12} {mb_7:<12.1f} {mb_21:<12.1f}")

        # Photometric Perturbations (Table X)
        if 'photometric' in subset_results:
            print("\n" + "=" * 100)
            print(f"TABLE X: PHOTOMETRIC PERTURBATIONS - {subset_name.upper()}")
            print("=" * 100)
            print(f"{'Perturbation':<20} {'Value':<12} {'Accuracy (%)':<15}")
            print("-" * 50)

            photometric = subset_results['photometric']
            for key, results in photometric.items():
                acc = results['accuracy'] * 100
                print(f"{key.replace('_', ' ').title():<20} {key.split('_')[-1]:<12} {acc:<15.1f}")

        # Adversarial Robustness
        if 'adversarial' in subset_results:
            print("\n" + "=" * 100)
            print(f"ADVERSARIAL ROBUSTNESS - {subset_name.upper()}")
            print("=" * 100)

            if 'fgsm' in subset_results['adversarial']:
                print("\n--- FGSM Attack ---")
                print(f"{'ε':<10} {'Accuracy (%)':<15}")
                print("-" * 30)
                for epsilon, results in subset_results['adversarial']['fgsm'].items():
                    acc = results['accuracy'] * 100
                    eps_val = int(epsilon * 255)
                    print(f"ε={eps_val}/255{'':<4} {acc:<15.1f}")

            if 'pgd' in subset_results['adversarial']:
                print("\n--- PGD-20 Attack ---")
                print(f"{'ε':<10} {'Accuracy (%)':<15}")
                print("-" * 30)
                for epsilon, results in subset_results['adversarial']['pgd'].items():
                    acc = results['accuracy'] * 100
                    eps_val = int(epsilon * 255)
                    print(f"ε={eps_val}/255{'':<4} {acc:<15.1f}")


if __name__ == "__main__":
    main()