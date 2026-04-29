import torch
import torch.nn as nn
from torchvision import transforms
import clip


class CLIPAppearanceModel(nn.Module):
    """CLIP-based appearance feature extractor"""

    def __init__(self, device='cuda'):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        )

    def forward(self, video_frames):
        """Extract appearance features from video frames"""
        batch_size, num_frames, C, H, W = video_frames.shape
        frames = video_frames.view(-1, C, H, W)

        frames = transforms.Resize(224)(frames)
        frames = transforms.CenterCrop(224)(frames)
        frames = self.normalize(frames)

        with torch.no_grad():
            features = self.clip_model.encode_image(frames.half())
        return features.view(batch_size, num_frames, -1).float()