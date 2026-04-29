import torch
import torch.nn as nn
from transformers import VideoMAEImageProcessor, VideoMAEModel


class VideoMAEMotionModel(nn.Module):
    """VideoMAE-based motion feature extractor"""

    def __init__(self, freeze_layers=8, device='cuda'):
        super().__init__()
        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.device = device

        for i, layer in enumerate(self.model.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, video_frames):
        """Extract motion features from video frames"""
        batch_pixel_values = []

        for video in video_frames:
            frame_list_np = [
                frame.permute(1, 2, 0).cpu().numpy() for frame in video
            ]

            inputs = self.processor(
                images=frame_list_np,
                return_tensors="pt",
                do_rescale=False
            ).to(self.device)

            batch_pixel_values.append(inputs.pixel_values)

        pixel_values = torch.cat(batch_pixel_values, dim=0)

        with torch.cuda.amp.autocast():
            outputs = self.model(pixel_values=pixel_values)
            features = outputs.last_hidden_state

        features = features.permute(0, 2, 1)
        pooled = self.temporal_pool(features).squeeze(-1)

        return pooled.float()