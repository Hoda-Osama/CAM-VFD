import torch
import yaml
from data import create_dataloaders
from models import CLIPAppearanceModel, EnhancedDepthModel, VideoMAEMotionModel, CrossAttentionFusion
from utils import MetricsCalculator, Visualizer, get_device


def evaluate_model(model, appearance_model, depth_model, motion_model, loader, device, title="Test"):
    """Evaluate model on given dataloader"""
    model.eval()
    appearance_model.eval()
    depth_model.eval()
    motion_model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in loader:
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
    MetricsCalculator.print_metrics(metrics, title)

    return metrics, all_probs, all_labels


def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = get_device()

    # Create dataloaders
    _, _, test_loader, _ = create_dataloaders(
        config['paths']['fake_videos'],
        config['paths']['real_videos'],
        config,
        device
    )

    # Initialize models
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
        dropout=config['model']['fusion']['dropout']
    ).to(device)

    # Load trained model
    fusion_model.load_state_dict(torch.load("best_model.pth"))

    # Evaluate
    test_metrics, test_probs, test_labels = evaluate_model(
        fusion_model, appearance_model, depth_model, motion_model,
        test_loader, device, "Test Set Results"
    )

    # Plot ROC curve
    visualizer = Visualizer()
    visualizer.plot_roc_curve(test_labels, test_probs, "ROC Curve - Test Set", "test_roc_curve.png")


if __name__ == "__main__":
    import numpy as np

    main()