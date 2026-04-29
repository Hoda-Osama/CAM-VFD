import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from models import CLIPAppearanceModel, EnhancedDepthModel, VideoMAEMotionModel, CrossAttentionFusion
from utils import get_device


class DeepfakeDetector:
    """Inference class for deepfake detection"""

    def __init__(self, model_path='best_model.pth', config_path='config/config.yaml'):
        self.device = get_device()
        self.transform = self._get_transform()

        # Initialize models
        self.appearance_model = CLIPAppearanceModel(device=self.device).to(self.device).eval()
        self.depth_model = EnhancedDepthModel().to(self.device).eval()
        self.motion_model = VideoMAEMotionModel(device=self.device).to(self.device).eval()

        self.fusion_model = CrossAttentionFusion(
            app_dim=512, motion_dim=768, depth_dim=128
        ).to(self.device)

        # Load trained weights
        self.fusion_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.fusion_model.eval()

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def load_frames(self, video_path, frame_count=16):
        """Load and preprocess frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < frame_count:
            indices = np.random.choice(total_frames, frame_count, replace=True)
        else:
            indices = np.linspace(0, total_frames - 1, frame_count, dtype=int)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((256, 256, 3), dtype=np.uint8)

            frame = cv2.resize(frame, (256, 256))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        if len(frames) < frame_count:
            frames.extend([torch.zeros_like(frames[0])] * (frame_count - len(frames)))

        return torch.stack(frames).unsqueeze(0)  # Add batch dimension

    @torch.no_grad()
    def predict(self, video_path):
        """Predict if video is real or fake"""
        frames = self.load_frames(video_path).to(self.device)

        # Extract features
        app_feats = self.appearance_model(frames)
        depth_feats = self.depth_model(frames)
        motion_feats = self.motion_model(frames)

        # Get prediction
        output = self.fusion_model(app_feats, depth_feats, motion_feats)
        probability = torch.sigmoid(output).item()
        prediction = "REAL" if probability > 0.5 else "FAKE"

        return {
            'prediction': prediction,
            'probability': probability if probability > 0.5 else 1 - probability,
            'raw_score': output.item()
        }


def main():
    # Example usage
    detector = DeepfakeDetector()

    # Test on a video
    video_path = "path/to/your/video.mp4"
    result = detector.predict(video_path)

    print(f"Video: {video_path}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['probability']:.4f}")


if __name__ == "__main__":
    main()