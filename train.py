import torch
import yaml
import logging
from data.dataset import load_multi_datasets
from models import CLIPAppearanceModel, EnhancedDepthModel, VideoMAEMotionModel, CrossAttentionFusion
from utils import Trainer, Visualizer, set_seed, get_device


def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Setup
    set_seed(42)
    device = get_device()
    logging.basicConfig(level=logging.INFO)

    print(f"Using device: {device}")
    print(f"Training for {config['training']['epochs']} epochs")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")

    # Create dataloaders from multiple datasets
    train_loader, val_loader, test_loader, dataset_stats = load_multi_datasets(config, device)

    print(f"\nDataset Statistics:")
    for dataset_name, stats in dataset_stats.items():
        print(f"\n  {dataset_name}:")
        for split, split_stats in stats.items():
            print(f"    {split.capitalize()}: Real={split_stats['real']}, Fake={split_stats['fake']}")

    # Initialize models
    print("\nInitializing models...")
    appearance_model = CLIPAppearanceModel(device=device).to(device).eval()
    depth_model = EnhancedDepthModel().to(device).eval()
    motion_model = VideoMAEMotionModel(
        freeze_layers=config['model']['motion']['freeze_layers'],
        device=device
    ).to(device).eval()

    fusion_model = CrossAttentionFusion(
        app_dim=config['model']['appearance']['feature_dim'],
        motion_dim=config['model']['motion']['feature_dim'],
        depth_dim=config['model']['depth']['feature_dim'],
        hidden_dim=config['model']['fusion']['hidden_dim'],
        num_heads=config['model']['fusion']['num_heads'],
        dropout=config['model']['fusion']['dropout'],
        cmad_enabled=config['model']['cmad']['enabled']
    ).to(device)

    # Train model
    trainer = Trainer(
        fusion_model, appearance_model, depth_model, motion_model,
        train_loader, val_loader, config, device,
        cmad_enabled=config['model']['cmad']['enabled']
    )

    trained_model, metrics = trainer.train()

    # Save training metrics
    torch.save(metrics, "training_metrics.pth")

    # Plot metrics
    visualizer = Visualizer()
    visualizer.plot_training_metrics(metrics, 'final', 'training_plots.png')

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")

    # Optional: Evaluate on each test set separately
    print("\nEvaluating on test sets...")
    evaluate_on_test_sets(trained_model, appearance_model, depth_model, motion_model,
                          test_loader, device, dataset_stats)


def evaluate_on_test_sets(model, appearance_model, depth_model, motion_model,
                          test_loader, device, dataset_stats):
    """Evaluate model on combined test set"""
    from utils import MetricsCalculator

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for frames, labels, _ in test_loader:
            frames = frames.to(device)

            app_feats = appearance_model(frames)
            depth_feats = depth_model(frames)
            motion_feats = motion_model(frames)

            outputs = model(app_feats, depth_feats, motion_feats)
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    predictions = (np.array(all_probs) > 0.5).astype(int)
    metrics = MetricsCalculator.calculate(predictions, all_labels, all_probs)
    MetricsCalculator.print_metrics(metrics, "Combined Test Set Results")


if __name__ == "__main__":
    import numpy as np

    main()