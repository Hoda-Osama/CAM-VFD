import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class EnhancedDepthModel(nn.Module):
    """MiDaS-based depth feature extractor"""

    def __init__(self):
        super().__init__()
        self.midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
        self.midas.eval()

        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.feature_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128)
        )

        for param in self.midas.parameters():
            param.requires_grad = False

    def forward(self, video_frames):
        """Extract depth features from video frames"""
        batch_size, num_frames, C, H, W = video_frames.shape
        features = []

        chunk_size = 4
        for i in range(0, num_frames, chunk_size):
            chunk = video_frames[:, i:i + chunk_size]
            chunk = chunk.reshape(-1, C, H, W)

            with torch.no_grad():
                normalized_chunk = self.normalize(chunk)
                resized = F.interpolate(normalized_chunk, size=(384, 384), mode='bilinear')
                depth = self.midas(resized)
                depth = depth.unsqueeze(1)
                depth = F.interpolate(depth, size=(H, W), mode='bilinear')

                N, _, H_d, W_d = depth.shape
                depth_flat = depth.view(N, -1)
                depth_flat = (depth_flat - depth_flat.mean(dim=1, keepdim=True)) / (
                        depth_flat.std(dim=1, keepdim=True) + 1e-6)
                depth = depth_flat.view(N, 1, H_d, W_d)

            chunk_features = self.feature_net(depth)
            chunk_features = chunk_features.view(batch_size, -1, 128)
            features.append(chunk_features)

        features = torch.cat(features, dim=1)
        return features.mean(dim=1)