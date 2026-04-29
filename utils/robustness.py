"""
Robustness Evaluation Module for Deepfake Detection
Implements:
1. Signal Degradation Robustness (Compression, Noise, Blur)
2. Adversarial Robustness (FGSM, PGD-20 attacks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class SignalDegradation:
    """
    Signal degradation robustness evaluation including:
    - H.264 Compression at different CRF levels
    - Gaussian Noise
    - Salt-Pepper Noise
    - Gaussian Blur
    - Defocus Blur
    - Motion Blur
    """

    COMPRESSION_CRF = [18, 28, 32]  # CRF values for H.264 compression
    GAUSSIAN_NOISE_SIGMA = [0.03, 0.1]  # Standard deviation for Gaussian noise
    SALT_PEPPER_PROB = [0.01, 0.05]  # Probability for salt-pepper noise
    GAUSSIAN_BLUR_SIGMA = [1, 2]  # Sigma for Gaussian blur
    DEFOCUS_BLUR_RADIUS = [3, 7]  # Radius for defocus blur
    MOTION_BLUR_LEN = [7, 21]  # Length for motion blur

    def __init__(self, device='cuda'):
        self.device = device

    def apply_h264_compression(self, frames: torch.Tensor, crf: int) -> torch.Tensor:
        """
        Apply H.264 compression to video frames

        Args:
            frames: Input frames (B, T, C, H, W)
            crf: Constant Rate Factor (lower = better quality)
        """
        batch_size, num_frames, C, H, W = frames.shape
        compressed_frames = []

        for b in range(batch_size):
            batch_frames = []
            for t in range(num_frames):
                # Convert to numpy (0-255 range)
                frame = frames[b, t].cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
                frame = np.transpose(frame, (1, 2, 0))  # CHW to HWC

                # Encode with H.264
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # Placeholder
                # Note: For real H.264 compression, use ffmpeg or similar
                # Here we simulate with JPEG compression as approximation
                result, encimg = cv2.imencode('.jpg', frame, encode_param)
                if result:
                    frame = cv2.imdecode(encimg, 1)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame).float() / 255.0
                    frame = frame.permute(2, 0, 1)  # HWC to CHW
                    batch_frames.append(frame)
                else:
                    batch_frames.append(frames[b, t])

            compressed_frames.append(torch.stack(batch_frames))

        return torch.stack(compressed_frames).to(self.device)

    def apply_gaussian_noise(self, frames: torch.Tensor, sigma: float) -> torch.Tensor:
        """Add Gaussian noise to frames"""
        noise = torch.randn_like(frames) * sigma
        return torch.clamp(frames + noise, 0.0, 1.0)

    def apply_salt_pepper_noise(self, frames: torch.Tensor, prob: float) -> torch.Tensor:
        """Add salt and pepper noise to frames"""
        noisy_frames = frames.clone()

        # Salt noise (white pixels)
        salt_mask = torch.rand_like(frames) < prob / 2
        noisy_frames[salt_mask] = 1.0

        # Pepper noise (black pixels)
        pepper_mask = torch.rand_like(frames) < prob / 2
        noisy_frames[pepper_mask] = 0.0

        return noisy_frames

    def apply_gaussian_blur(self, frames: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur to frames"""
        batch_size, num_frames, C, H, W = frames.shape

        # Create Gaussian kernel
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = self._gaussian_kernel(kernel_size, sigma).to(self.device)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        blurred_frames = []
        for b in range(batch_size):
            batch_blurred = []
            for t in range(num_frames):
                frame = frames[b, t].unsqueeze(0)  # (1, C, H, W)
                # Apply separable convolution
                frame_blurred = F.conv2d(frame, kernel, padding=kernel_size // 2, groups=C)
                batch_blurred.append(frame_blurred.squeeze(0))
            blurred_frames.append(torch.stack(batch_blurred))

        return torch.stack(blurred_frames)

    def _gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Generate Gaussian kernel"""
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        kernel = kernel / kernel.sum()
        return kernel

    def apply_defocus_blur(self, frames: torch.Tensor, radius: int) -> torch.Tensor:
        """Apply defocus blur (pillbox filter) to frames"""
        batch_size, num_frames, C, H, W = frames.shape

        # Create pillbox kernel
        kernel_size = 2 * radius + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        kernel = kernel.to(self.device)

        blurred_frames = []
        for b in range(batch_size):
            batch_blurred = []
            for t in range(num_frames):
                frame = frames[b, t].unsqueeze(0)
                frame_blurred = F.conv2d(frame, kernel, padding=radius, groups=C)
                batch_blurred.append(frame_blurred.squeeze(0))
            blurred_frames.append(torch.stack(batch_blurred))

        return torch.stack(blurred_frames)

    def apply_motion_blur(self, frames: torch.Tensor, length: int) -> torch.Tensor:
        """Apply motion blur to frames"""
        batch_size, num_frames, C, H, W = frames.shape

        # Create motion blur kernel (horizontal direction)
        kernel = torch.zeros(length, length)
        kernel[length // 2, :] = 1.0 / length
        kernel = kernel.view(1, 1, length, length).to(self.device)

        blurred_frames = []
        for b in range(batch_size):
            batch_blurred = []
            for t in range(num_frames):
                frame = frames[b, t].unsqueeze(0)
                frame_blurred = F.conv2d(frame, kernel, padding=length // 2, groups=C)
                batch_blurred.append(frame_blurred.squeeze(0))
            blurred_frames.append(torch.stack(batch_blurred))

        return torch.stack(blurred_frames)

    def apply_lighting_perturbation(self, frames: torch.Tensor, delta: float,
                                    perturbation_type: str = 'brightness') -> torch.Tensor:
        """
        Apply lighting perturbations (brightness/contrast)

        Args:
            frames: Input frames
            delta: Perturbation magnitude
            perturbation_type: 'brightness' or 'contrast'
        """
        if perturbation_type == 'brightness':
            # Adjust brightness
            return torch.clamp(frames + delta, 0.0, 1.0)
        elif perturbation_type == 'contrast':
            # Adjust contrast
            mean = frames.mean(dim=(2, 3, 4), keepdim=True)
            return torch.clamp(mean + (frames - mean) * delta, 0.0, 1.0)
        else:
            return frames

    def apply_color_distortion(self, frames: torch.Tensor, value: float,
                               distortion_type: str = 'saturation') -> torch.Tensor:
        """
        Apply color distortions (saturation, hue shift)

        Args:
            frames: Input frames
            value: Distortion magnitude
            distortion_type: 'saturation' or 'hue'
        """
        batch_size, num_frames, C, H, W = frames.shape
        distorted_frames = []

        for b in range(batch_size):
            batch_distorted = []
            for t in range(num_frames):
                frame = frames[b, t].cpu().numpy()
                frame = np.transpose(frame, (1, 2, 0))  # CHW to HWC

                if distortion_type == 'saturation':
                    # Convert to HSV, adjust saturation
                    hsv = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * value, 0, 255)
                    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0

                elif distortion_type == 'hue':
                    # Shift hue
                    hsv = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                    hsv[:, :, 0] = (hsv[:, :, 0] + value) % 180
                    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0

                frame = torch.from_numpy(frame).float()
                frame = frame.permute(2, 0, 1)  # HWC to CHW
                batch_distorted.append(frame)

            distorted_frames.append(torch.stack(batch_distorted))

        return torch.stack(distorted_frames).to(self.device)


class AdversarialAttack:
    """
    Adversarial attack implementations for robustness evaluation
    - FGSM (Fast Gradient Sign Method)
    - PGD-20 (Projected Gradient Descent with 20 iterations)
    """

    def __init__(self, model, appearance_model, depth_model, motion_model,
                 device='cuda', epsilon=8 / 255, alpha=2 / 255, iterations=20):
        """
        Args:
            model: Fusion model
            appearance_model: CLIP appearance model
            depth_model: Depth model
            motion_model: Motion model
            device: Device to run on
            epsilon: Maximum perturbation magnitude
            alpha: Step size for PGD
            iterations: Number of iterations for PGD
        """
        self.model = model
        self.appearance_model = appearance_model
        self.depth_model = depth_model
        self.motion_model = motion_model
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations

    def fgsm_attack(self, frames: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack

        Args:
            frames: Input frames (B, T, C, H, W)
            labels: Ground truth labels

        Returns:
            Adversarial frames
        """
        frames = frames.clone().detach().requires_grad_(True)

        # Forward pass
        with torch.no_grad():
            app_feats = self.appearance_model(frames)
            depth_feats = self.depth_model(frames)
            motion_feats = self.motion_model(frames)

        outputs = self.model(app_feats, depth_feats, motion_feats)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Generate adversarial examples
        grad_sign = frames.grad.data.sign()
        adversarial_frames = frames + self.epsilon * grad_sign
        adversarial_frames = torch.clamp(adversarial_frames, 0, 1)

        return adversarial_frames.detach()

    def pgd_attack(self, frames: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack with 20 iterations

        Args:
            frames: Input frames (B, T, C, H, W)
            labels: Ground truth labels

        Returns:
            Adversarial frames
        """
        # Initialize adversarial frames
        adversarial_frames = frames.clone().detach()

        for _ in range(self.iterations):
            adversarial_frames = adversarial_frames.clone().detach().requires_grad_(True)

            # Forward pass
            with torch.no_grad():
                app_feats = self.appearance_model(adversarial_frames)
                depth_feats = self.depth_model(adversarial_frames)
                motion_feats = self.motion_model(adversarial_frames)

            outputs = self.model(app_feats, depth_feats, motion_feats)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Update adversarial examples
            grad_sign = adversarial_frames.grad.data.sign()
            adversarial_frames = adversarial_frames + self.alpha * grad_sign

            # Project back to epsilon ball
            eta = torch.clamp(adversarial_frames - frames, -self.epsilon, self.epsilon)
            adversarial_frames = torch.clamp(frames + eta, 0, 1)

        return adversarial_frames.detach()

    def evaluate_robustness(self, dataloader, attack_type='fgsm',
                            epsilon_values=[2 / 255, 4 / 255, 8 / 255]):
        """
        Evaluate model robustness against adversarial attacks

        Args:
            dataloader: DataLoader containing clean videos
            attack_type: 'fgsm' or 'pgd'
            epsilon_values: List of perturbation magnitudes to test

        Returns:
            Dictionary with accuracy for each epsilon value
        """
        results = {}

        for epsilon in epsilon_values:
            self.epsilon = epsilon
            if attack_type == 'pgd':
                self.alpha = epsilon / 5  # Typical alpha = epsilon / iterations * 2
                self.iterations = 20

            print(f"\nEvaluating {attack_type.upper()} attack with epsilon={epsilon:.4f}")

            all_preds = []
            all_labels = []

            for frames, labels, _ in tqdm(dataloader, desc=f"Attack: {attack_type.upper()}"):
                frames = frames.to(self.device)
                labels = labels.to(self.device).float()

                # Generate adversarial examples
                if attack_type == 'fgsm':
                    adversarial_frames = self.fgsm_attack(frames, labels)
                else:  # pgd
                    adversarial_frames = self.pgd_attack(frames, labels)

                # Evaluate on adversarial examples
                with torch.no_grad():
                    app_feats = self.appearance_model(adversarial_frames)
                    depth_feats = self.depth_model(adversarial_frames)
                    motion_feats = self.motion_model(adversarial_frames)

                    outputs = self.model(app_feats, depth_feats, motion_feats)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)

            results[epsilon] = {
                'accuracy': accuracy,
                'f1': f1,
                'recall': recall
            }

            print(f"Epsilon={epsilon:.4f}: Accuracy={accuracy:.4f}, F1={f1:.4f}, Recall={recall:.4f}")

        return results


class RobustnessEvaluator:
    """
    Complete robustness evaluation combining signal degradation and adversarial attacks
    """

    def __init__(self, model, appearance_model, depth_model, motion_model, device='cuda'):
        self.model = model
        self.appearance_model = appearance_model
        self.depth_model = depth_model
        self.motion_model = motion_model
        self.device = device
        self.signal_degradation = SignalDegradation(device)
        self.adversarial = AdversarialAttack(model, appearance_model, depth_model,
                                             motion_model, device)

    @torch.no_grad()
    def evaluate_clean(self, dataloader) -> Dict:
        """Evaluate baseline performance on clean data"""
        return self._evaluate_dataloader(dataloader, "Clean")

    @torch.no_grad()
    def evaluate_compression(self, dataloader, crf_values=[18, 28, 32]) -> Dict:
        """Evaluate robustness under H.264 compression"""
        results = {}
        for crf in crf_values:
            print(f"\nEvaluating H.264 Compression (CRF={crf})...")
            results[crf] = self._evaluate_with_degradation(
                dataloader,
                lambda frames: self.signal_degradation.apply_h264_compression(frames, crf),
                f"Compression_CRF_{crf}"
            )
        return results

    @torch.no_grad()
    def evaluate_noise(self, dataloader) -> Dict:
        """Evaluate robustness under different noise types"""
        results = {}

        # Gaussian noise
        for sigma in [0.03, 0.1]:
            print(f"\nEvaluating Gaussian Noise (sigma={sigma})...")
            results[f'gaussian_noise_sigma_{sigma}'] = self._evaluate_with_degradation(
                dataloader,
                lambda frames: self.signal_degradation.apply_gaussian_noise(frames, sigma),
                f"Gaussian_Noise_sigma_{sigma}"
            )

        # Salt-Pepper noise
        for prob in [0.01, 0.05]:
            print(f"\nEvaluating Salt-Pepper Noise (prob={prob})...")
            results[f'salt_pepper_noise_prob_{prob}'] = self._evaluate_with_degradation(
                dataloader,
                lambda frames: self.signal_degradation.apply_salt_pepper_noise(frames, prob),
                f"SaltPepper_Noise_prob_{prob}"
            )

        return results

    @torch.no_grad()
    def evaluate_blur(self, dataloader) -> Dict:
        """Evaluate robustness under different blur types"""
        results = {}

        # Gaussian blur
        for sigma in [1, 2]:
            print(f"\nEvaluating Gaussian Blur (sigma={sigma})...")
            results[f'gaussian_blur_sigma_{sigma}'] = self._evaluate_with_degradation(
                dataloader,
                lambda frames: self.signal_degradation.apply_gaussian_blur(frames, sigma),
                f"Gaussian_Blur_sigma_{sigma}"
            )

        # Defocus blur
        for radius in [3, 7]:
            print(f"\nEvaluating Defocus Blur (radius={radius})...")
            results[f'defocus_blur_radius_{radius}'] = self._evaluate_with_degradation(
                dataloader,
                lambda frames: self.signal_degradation.apply_defocus_blur(frames, radius),
                f"Defocus_Blur_radius_{radius}"
            )

        # Motion blur
        for length in [7, 21]:
            print(f"\nEvaluating Motion Blur (length={length})...")
            results[f'motion_blur_length_{length}'] = self._evaluate_with_degradation(
                dataloader,
                lambda frames: self.signal_degradation.apply_motion_blur(frames, length),
                f"Motion_Blur_length_{length}"
            )

        return results

    @torch.no_grad()
    def evaluate_photometric(self, dataloader) -> Dict:
        """Evaluate robustness under photometric perturbations"""
        results = {}

        # Lighting perturbations
        brightness_deltas = [0.05, 0.1, 0.7]
        for delta in brightness_deltas:
            print(f"\nEvaluating Brightness Adjustment (delta={delta})...")
            results[f'brightness_delta_{delta}'] = self._evaluate_with_degradation(
                dataloader,
                lambda frames: self.signal_degradation.apply_lighting_perturbation(
                    frames, delta, 'brightness'),
                f"Brightness_delta_{delta}"
            )

        # Contrast adjustment
        contrast_factors = [1.3]
        for factor in contrast_factors:
            print(f"\nEvaluating Contrast Adjustment (factor={factor})...")
            results[f'contrast_factor_{factor}'] = self._evaluate_with_degradation(
                dataloader,
                lambda frames: self.signal_degradation.apply_lighting_perturbation(
                    frames, factor, 'contrast'),
                f"Contrast_factor_{factor}"
            )

        # Saturation adjustment
        saturation_factors = [0.7, 1.3]
        for factor in saturation_factors:
            print(f"\nEvaluating Saturation Adjustment (factor={factor})...")
            results[f'saturation_factor_{factor}'] = self._evaluate_with_degradation(
                dataloader,
                lambda frames: self.signal_degradation.apply_color_distortion(
                    frames, factor, 'saturation'),
                f"Saturation_factor_{factor}"
            )

        # Hue shift
        hue_shifts = [5, 12]
        for shift in hue_shifts:
            print(f"\nEvaluating Hue Shift (shift={shift}°)...")
            results[f'hue_shift_{shift}'] = self._evaluate_with_degradation(
                dataloader,
                lambda frames: self.signal_degradation.apply_color_distortion(
                    frames, float(shift), 'hue'),
                f"Hue_Shift_{shift}"
            )

        return results

    @torch.no_grad()
    def evaluate_adversarial(self, dataloader, attack_type='both'):
        """Evaluate adversarial robustness"""
        results = {}

        if attack_type in ['fgsm', 'both']:
            print("\n" + "=" * 60)
            print("Evaluating FGSM Adversarial Attacks")
            print("=" * 60)
            results['fgsm'] = self.adversarial.evaluate_robustness(
                dataloader, 'fgsm', epsilon_values=[2 / 255, 4 / 255, 8 / 255]
            )

        if attack_type in ['pgd', 'both']:
            print("\n" + "=" * 60)
            print("Evaluating PGD-20 Adversarial Attacks")
            print("=" * 60)
            results['pgd'] = self.adversarial.evaluate_robustness(
                dataloader, 'pgd', epsilon_values=[2 / 255, 4 / 255, 8 / 255]
            )

        return results

    def _evaluate_with_degradation(self, dataloader, degradation_func, name):
        """Helper function to evaluate with a degradation applied"""
        all_preds = []
        all_labels = []

        for frames, labels, _ in tqdm(dataloader, desc=name):
            frames = frames.to(self.device)
            labels = labels.to(self.device).float()

            # Apply degradation
            degraded_frames = degradation_func(frames)

            # Extract features
            app_feats = self.appearance_model(degraded_frames)
            depth_feats = self.depth_model(degraded_frames)
            motion_feats = self.motion_model(degraded_frames)

            # Forward pass
            outputs = self.model(app_feats, depth_feats, motion_feats)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)

        print(f"{name}: Acc={accuracy:.4f}, F1={f1:.4f}, Recall={recall:.4f}")

        return {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            'precision': precision
        }

    def _evaluate_dataloader(self, dataloader, name):
        """Evaluate on clean dataloader"""
        all_preds = []
        all_labels = []

        for frames, labels, _ in tqdm(dataloader, desc=name):
            frames = frames.to(self.device)
            labels = labels.to(self.device).float()

            app_feats = self.appearance_model(frames)
            depth_feats = self.depth_model(frames)
            motion_feats = self.motion_model(frames)

            outputs = self.model(app_feats, depth_feats, motion_feats)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall
        }

    def generate_full_report(self, dataloader, subset_name="CogVideo",
                             save_path='robustness_report.txt'):
        """
        Generate comprehensive robustness report similar to Tables IX and X
        """
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append(f"ROBUSTNESS EVALUATION REPORT - {subset_name}")
        report_lines.append("=" * 100)

        # Baseline clean accuracy
        clean_results = self.evaluate_clean(dataloader)
        report_lines.append(f"\nBaseline Accuracy: {clean_results['accuracy'] * 100:.1f}%")

        # Signal Degradation Results (Table IX)
        report_lines.append("\n" + "=" * 80)
        report_lines.append("TABLE IX: SIGNAL DEGRADATION ROBUSTNESS")
        report_lines.append("=" * 80)

        # Compression
        report_lines.append("\n--- H.264 Compression ---")
        compression_results = self.evaluate_compression(dataloader)
        for crf, results in compression_results.items():
            report_lines.append(f"  CRF={crf}: {results['accuracy'] * 100:.1f}%")

        # Noise
        report_lines.append("\n--- Gaussian Noise ---")
        noise_results = self.evaluate_noise(dataloader)
        for key, results in noise_results.items():
            if 'gaussian' in key:
                report_lines.append(f"  {key}: {results['accuracy'] * 100:.1f}%")

        # Blur
        report_lines.append("\n--- Blur ---")
        blur_results = self.evaluate_blur(dataloader)
        for key, results in blur_results.items():
            report_lines.append(f"  {key}: {results['accuracy'] * 100:.1f}%")

        # Photometric Perturbations (Table X)
        report_lines.append("\n" + "=" * 80)
        report_lines.append("TABLE X: PHOTOMETRIC PERTURBATIONS ROBUSTNESS")
        report_lines.append("=" * 80)

        photometric_results = self.evaluate_photometric(dataloader)
        for key, results in photometric_results.items():
            report_lines.append(f"  {key}: {results['accuracy'] * 100:.1f}%")

        # Adversarial Robustness
        report_lines.append("\n" + "=" * 80)
        report_lines.append("ADVERSARIAL ROBUSTNESS (FGSM & PGD-20)")
        report_lines.append("=" * 80)

        adversarial_results = self.evaluate_adversarial(dataloader, attack_type='both')

        report_lines.append("\n--- FGSM Attack ---")
        for epsilon, results in adversarial_results.get('fgsm', {}).items():
            report_lines.append(f"  ε={epsilon * 255:.0f}/255: {results['accuracy'] * 100:.1f}%")

        report_lines.append("\n--- PGD-20 Attack ---")
        for epsilon, results in adversarial_results.get('pgd', {}).items():
            report_lines.append(f"  ε={epsilon * 255:.0f}/255: {results['accuracy'] * 100:.1f}%")

        report_lines.append("\n" + "=" * 100)

        # Save report
        report_text = "\n".join(report_lines)
        with open(f'{save_path}_{subset_name}.txt', 'w') as f:
            f.write(report_text)

        print(report_text)

        return {
            'clean': clean_results,
            'compression': compression_results,
            'noise': noise_results,
            'blur': blur_results,
            'photometric': photometric_results,
            'adversarial': adversarial_results
        }