import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


class Visualizer:
    """Visualization utilities for training and evaluation"""

    @staticmethod
    def plot_training_metrics(metrics, epoch, save_path='training_plots.png'):
        """Plot training metrics over epochs"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(metrics['train_loss'], label='Train Loss', marker='o')
        axes[0, 0].plot(metrics['val_loss'], label='Val Loss', marker='s')
        axes[0, 0].set_title('Loss over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy
        axes[0, 1].plot(metrics['train_acc'], label='Train Acc', marker='o')
        axes[0, 1].plot(metrics['val_acc'], label='Val Acc', marker='s')
        axes[0, 1].set_title('Accuracy over Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # F1-Score
        axes[0, 2].plot(metrics['train_f1'], label='Train F1', marker='o')
        axes[0, 2].plot(metrics['val_f1'], label='Val F1', marker='s')
        axes[0, 2].set_title('F1-Score over Epochs')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1-Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Recall
        axes[1, 0].plot(metrics['train_recall'], label='Train Recall', marker='o')
        axes[1, 0].plot(metrics['val_recall'], label='Val Recall', marker='s')
        axes[1, 0].set_title('Recall over Epochs')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # AUROC
        if 'train_auroc' in metrics:
            axes[1, 1].plot(metrics['train_auroc'], label='Train AUROC', marker='o')
            axes[1, 1].plot(metrics['val_auroc'], label='Val AUROC', marker='s')
            axes[1, 1].set_title('AUROC over Epochs')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AUROC')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        # Learning Rate
        axes[1, 2].plot(metrics['learning_rate'], label='Learning Rate', color='green', marker='o')
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        plt.suptitle(f'Training Metrics - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_roc_curve(labels, probabilities, title="ROC Curve", save_path='roc_curve.png'):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auroc = roc_auc_score(labels, probabilities)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def visualize_sampling_strategy(video_length, frame_count, sampling_indices,
                                    save_path='sampling_visualization.png'):
        """Visualize the adaptive sampling strategy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot video timeline
        timeline = np.zeros(video_length)
        timeline[sampling_indices] = 1

        ax1.bar(range(video_length), timeline, width=0.8, color='blue', alpha=0.7)
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Selected')
        ax1.set_title(f'Frame Selection Pattern\n{len(sampling_indices)} frames from {video_length} total')
        ax1.set_ylim(0, 1.2)

        # Plot consecutive segments
        if len(sampling_indices) > 1:
            diffs = np.diff(sampling_indices)
            segments = []
            current_segment = [sampling_indices[0]]

            for i, diff in enumerate(diffs):
                if diff == 1:
                    current_segment.append(sampling_indices[i + 1])
                else:
                    if len(current_segment) > 1:
                        segments.append(current_segment)
                    current_segment = [sampling_indices[i + 1]]
            if len(current_segment) > 1:
                segments.append(current_segment)

            colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
            for seg, color in zip(segments, colors):
                ax2.plot(seg, [1] * len(seg), 'o-', color=color, markersize=8, linewidth=2)

        ax2.set_xlabel('Frame Index')
        ax2.set_yticks([])
        ax2.set_title('Consecutive Frame Segments')
        ax2.set_ylim(0.8, 1.2)

        plt.suptitle(f'Adaptive Sampling Strategy - Video Length: {video_length}')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()