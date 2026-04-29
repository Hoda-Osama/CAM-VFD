import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import logging


class CMADAnalyzer:
    """
    Comprehensive analyzer for Cross-Modal Attention Discrepancy (CMAD)
    Provides statistical analysis and visualization of CMAD distributions
    """

    def __init__(self, model, appearance_model, depth_model, motion_model, device):
        self.model = model
        self.appearance_model = appearance_model
        self.depth_model = depth_model
        self.motion_model = motion_model
        self.device = device

    @torch.no_grad()
    def compute_cmad_scores(self, dataloader, dataset_name="Unknown"):
        """
        Compute CMAD scores for all videos in a dataloader

        Args:
            dataloader: DataLoader containing videos
            dataset_name: Name of the dataset for logging

        Returns:
            cmad_scores: Array of CMAD scores
            labels: Array of ground truth labels
        """
        self.model.eval()
        self.appearance_model.eval()
        self.depth_model.eval()
        self.motion_model.eval()

        all_cmad_scores = []
        all_labels = []

        for frames, labels, _ in tqdm(dataloader, desc=f"Computing CMAD - {dataset_name}"):
            frames = frames.to(self.device)

            # Extract features
            app_feats = self.appearance_model(frames)
            depth_feats = self.depth_model(frames)
            motion_feats = self.motion_model(frames)

            # Get CMAD scores
            cmad_scores = self.model.get_cmad_score(app_feats, depth_feats, motion_feats)

            all_cmad_scores.extend(cmad_scores.cpu().numpy())
            all_labels.extend(labels.numpy())

        return np.array(all_cmad_scores), np.array(all_labels)

    def statistical_analysis(self, cmad_real, cmad_fake):
        """
        Perform statistical analysis on CMAD distributions

        Args:
            cmad_real: CMAD scores for real videos
            cmad_fake: CMAD scores for fake videos

        Returns:
            dict: Statistical metrics including t-test, Cohen's d, etc.
        """
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(cmad_real, cmad_fake)

        # Cohen's d (effect size)
        pooled_std = np.sqrt((np.var(cmad_real) + np.var(cmad_fake)) / 2)
        cohens_d = (np.mean(cmad_real) - np.mean(cmad_fake)) / pooled_std

        # Descriptive statistics
        stats_dict = {
            'real_mean': np.mean(cmad_real),
            'real_std': np.std(cmad_real),
            'real_median': np.median(cmad_real),
            'real_q1': np.percentile(cmad_real, 25),
            'real_q3': np.percentile(cmad_real, 75),
            'fake_mean': np.mean(cmad_fake),
            'fake_std': np.std(cmad_fake),
            'fake_median': np.median(cmad_fake),
            'fake_q1': np.percentile(cmad_fake, 25),
            'fake_q3': np.percentile(cmad_fake, 75),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.001
        }

        return stats_dict

    def plot_cmad_distribution(self, cmad_real, cmad_fake, save_path='cmad_distribution.png'):
        """
        Plot CMAD distribution comparison between real and fake videos
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Histogram
        axes[0].hist(cmad_real, bins=50, alpha=0.7, label='Real', color='green', density=True)
        axes[0].hist(cmad_fake, bins=50, alpha=0.7, label='Fake', color='red', density=True)
        axes[0].set_xlabel('CMAD Score')
        axes[0].set_ylabel('Density')
        axes[0].set_title('CMAD Distribution Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        box_data = [cmad_real, cmad_fake]
        bp = axes[1].boxplot(box_data, labels=['Real', 'Fake'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('salmon')
        axes[1].set_ylabel('CMAD Score')
        axes[1].set_title('CMAD Box Plot')
        axes[1].grid(True, alpha=0.3)

        # Violin plot
        violin_data = [cmad_real, cmad_fake]
        parts = axes[2].violinplot(violin_data, positions=[0, 1], showmeans=True, showmedians=True)
        parts['bodies'][0].set_facecolor('lightgreen')
        parts['bodies'][1].set_facecolor('salmon')
        axes[2].set_xticks([0, 1])
        axes[2].set_xticklabels(['Real', 'Fake'])
        axes[2].set_ylabel('CMAD Score')
        axes[2].set_title('CMAD Violin Plot')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle('Cross-Modal Attention Discrepancy (CMAD) Analysis')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_cmad_roc_curve(self, cmad_scores, labels, save_path='cmad_roc_curve.png'):
        """
        Plot ROC curve for CMAD-based classification
        """
        fpr, tpr, thresholds = roc_curve(labels, cmad_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'CMAD ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - CMAD-based Detection')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=150)
        plt.close()

        return roc_auc

    def plot_attention_heatmaps(self, frames, save_path='attention_heatmaps.png'):
        """
        Visualize attention heatmaps for cross-modal interactions
        """
        self.model.eval()

        with torch.no_grad():
            frames = frames.to(self.device)
            app_feats = self.appearance_model(frames)
            depth_feats = self.depth_model(frames)
            motion_feats = self.motion_model(frames)

            # Get attention weights
            _, _, attention_data = self.model(app_feats, depth_feats, motion_feats, return_cmad=True)

            motion_attention = attention_data['app_motion_attention'].cpu().numpy()
            depth_attention = attention_data['app_depth_attention'].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Appearance-motion attention
        im1 = axes[0].imshow(motion_attention[0].T, aspect='auto', cmap='hot')
        axes[0].set_title('Appearance-Motion Attention')
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Attention Heads')
        plt.colorbar(im1, ax=axes[0])

        # Appearance-depth attention
        im2 = axes[1].imshow(depth_attention[0].T, aspect='auto', cmap='hot')
        axes[1].set_title('Appearance-Depth Attention')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Attention Heads')
        plt.colorbar(im2, ax=axes[1])

        # Discrepancy heatmap
        discrepancy = np.abs(motion_attention[0] - depth_attention[0])
        im3 = axes[2].imshow(discrepancy.T, aspect='auto', cmap='coolwarm')
        axes[2].set_title('Attention Discrepancy')
        axes[2].set_xlabel('Time Steps')
        axes[2].set_ylabel('Attention Heads')
        plt.colorbar(im3, ax=axes[2])

        plt.suptitle('Cross-Modal Attention Heatmaps')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def generate_report(self, cmad_real, cmad_fake, dataset_name="GenVideo"):
        """
        Generate comprehensive CMAD analysis report
        """
        stats = self.statistical_analysis(cmad_real, cmad_fake)

        report = f"""
        {'=' * 60}
        CMAD Analysis Report - {dataset_name}
        {'=' * 60}

        REAL VIDEOS:
          Mean CMAD: {stats['real_mean']:.6f} ± {stats['real_std']:.6f}
          Median CMAD: {stats['real_median']:.6f}
          IQR: [{stats['real_q1']:.6f}, {stats['real_q3']:.6f}]

        FAKE VIDEOS:
          Mean CMAD: {stats['fake_mean']:.6f} ± {stats['fake_std']:.6f}
          Median CMAD: {stats['fake_median']:.6f}
          IQR: [{stats['fake_q1']:.6f}, {stats['fake_q3']:.6f}]

        STATISTICAL TESTS:
          T-statistic: {stats['t_statistic']:.4f}
          P-value: {stats['p_value']:.6e}
          Cohen's d: {stats['cohens_d']:.4f}
          Statistically Significant: {stats['significant']}

        {'=' * 60}
        """

        logging.info(report)

        # Save report to file
        with open(f'cmad_report_{dataset_name}.txt', 'w') as f:
            f.write(report)

        return stats

    def analyze_dataset(self, dataloader, dataset_name="GenVideo"):
        """
        Complete CMAD analysis for a dataset
        """
        print(f"\nAnalyzing CMAD for {dataset_name}...")
        cmad_scores, labels = self.compute_cmad_scores(dataloader, dataset_name)

        cmad_real = cmad_scores[labels == 1]
        cmad_fake = cmad_scores[labels == 0]

        # Generate plots
        self.plot_cmad_distribution(cmad_real, cmad_fake, f'cmad_distribution_{dataset_name}.png')
        roc_auc = self.plot_cmad_roc_curve(cmad_scores, labels, f'cmad_roc_{dataset_name}.png')

        # Generate statistical report
        stats = self.generate_report(cmad_real, cmad_fake, dataset_name)
        stats['roc_auc'] = roc_auc

        return stats, cmad_scores, labels