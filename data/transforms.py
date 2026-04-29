import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


def get_transforms(mode='train'):
    """
    Get image transforms with enhanced augmentations for training.
    Designed to prevent memorization of pixel patterns and lighting artifacts.

    Args:
        mode: 'train', 'val', or 'test'

    Returns:
        transforms.Compose object
    """
    if mode == 'train':
        return transforms.Compose([
            # Simulate different zoom levels and camera framing
            transforms.Resize(256),
            transforms.RandomResizedCrop(
                224,
                scale=(0.7, 1.0),  # Wider zoom range (70-100%)
                ratio=(0.9, 1.1)  # Slight aspect ratio variation
            ),

            # Simulate different camera orientations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),  # Rare vertical flips for robustness

            # Enhance resilience to lighting and color variations
            transforms.ColorJitter(
                brightness=0.25,  # Simulate different lighting conditions
                contrast=0.25,  # Vary contrast
                saturation=0.15,  # Slight color variation
                hue=0.05  # Small hue shifts
            ),

            # Simulate different camera sensor noises
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),

            # Random grayscale conversion with low probability
            transforms.RandomGrayscale(p=0.05),

            # Convert to tensor
            transforms.ToTensor(),

            # Random noise injection (helps prevent memorization)
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02 if torch.rand(1) > 0.8 else x),

            # Clip values to valid range
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])

    else:  # Validation and test transforms (minimal augmentation)
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])


# Alternative stronger augmentations for more robust training
def get_strong_transforms(mode='train'):
    """
    Stronger augmentation policy for more challenging training scenarios.
    Use this when the model shows signs of overfitting.
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.1),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 1.0)),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomRotation(degrees=5),  # Small rotations for robustness
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.03 if torch.rand(1) > 0.7 else x),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])


# Adaptive augmentation based on training progress
class AdaptiveAugmentation:
    """
    Dynamically adjusts augmentation strength based on training progress.
    Starts with weak augmentations and gradually increases strength.
    """

    def __init__(self, total_epochs, start_strong_epoch=5):
        self.total_epochs = total_epochs
        self.start_strong_epoch = start_strong_epoch

    def get_transform(self, current_epoch, mode='train'):
        if mode != 'train':
            return get_transforms('val')

        if current_epoch < self.start_strong_epoch:
            return get_transforms('train')  # Standard augmentation
        else:
            progress = min(1.0, (current_epoch - self.start_strong_epoch) / self.total_epochs)
            if progress < 0.5:
                return get_transforms('train')  # Keep standard
            else:
                return get_strong_transforms('train')  # Switch to strong