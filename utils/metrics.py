import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class MetricsCalculator:
    """Calculate and store evaluation metrics"""

    @staticmethod
    def calculate(predictions, labels, probabilities=None):
        """Calculate all metrics"""
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'precision': precision_score(labels, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, predictions)
        }

        if probabilities is not None:
            try:
                metrics['auroc'] = roc_auc_score(labels, probabilities)
            except:
                metrics['auroc'] = 0.0

        return metrics

    @staticmethod
    def print_metrics(metrics, title="Results"):
        """Print metrics in a formatted way"""
        print(f"\n{'=' * 50}")
        print(f"{title}")
        print(f"{'=' * 50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        if 'auroc' in metrics:
            print(f"AUROC:     {metrics['auroc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print(f"{'=' * 50}\n")