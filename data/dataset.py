import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from sklearn.model_selection import train_test_split
from .transforms import get_transforms


def get_video_files(folder_path, extensions=['.mp4', '.avi', '.mov']):
    """Get all video files from folder recursively"""
    videos = []
    for ext in extensions:
        videos.extend(glob.glob(os.path.join(folder_path, "**", f"*{ext}"), recursive=True))
    return videos


def load_dataset(fake_dir, real_dir):
    """Load fake and real video paths"""
    fake_videos = get_video_files(fake_dir)
    real_videos = get_video_files(real_dir)
    return fake_videos, real_videos


class VideoDataset(Dataset):
    """Video dataset with adaptive consecutive frame sampling"""

    def __init__(self, real_video_paths, fake_video_paths, transform=None,
                 frame_count=16, mode='train', min_frames=2, n_factor=3,
                 dataset_name="unknown"):
        """
        Args:
            real_video_paths: Paths to real videos
            fake_video_paths: Paths to fake videos
            transform: Optional transform to be applied
            frame_count: Number of frames to sample from each video (T)
            mode: 'train', 'val', or 'test'
            min_frames: Minimum number of frames to consider a video valid
            n_factor: Factor for determining medium vs long videos (n)
            dataset_name: Name of the dataset (for tracking)
        """
        self.video_paths = real_video_paths + fake_video_paths
        self.labels = [1] * len(real_video_paths) + [0] * len(fake_video_paths)
        self.dataset_names = [dataset_name] * len(self.video_paths)
        self.transform = transform
        self.frame_count = frame_count
        self.mode = mode
        self.min_frames = min_frames
        self.n_factor = n_factor
        self.valid_indices = self._filter_valid_videos()

        print(f"[{dataset_name}] Initialized with {len(self.valid_indices)}/{len(self.video_paths)} valid videos")

    def _filter_valid_videos(self):
        """Filter videos that can be opened and have enough frames"""
        valid_indices = []
        for idx in range(len(self.video_paths)):
            if self._validate_video(self.video_paths[idx]):
                valid_indices.append(idx)
        return valid_indices

    def _validate_video(self, path):
        """Check if video is valid and has enough frames"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count >= self.min_frames

    def __len__(self):
        return len(self.valid_indices)

    def get_dataset_info(self, idx):
        """Get dataset name for a given index"""
        actual_idx = self.valid_indices[idx]
        return self.dataset_names[actual_idx]

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        video_path = self.video_paths[actual_idx]
        label = self.labels[actual_idx]
        dataset_name = self.dataset_names[actual_idx]

        try:
            frames = self._load_frames_adaptive(video_path)
            frames = torch.stack(frames)
        except Exception as e:
            print(f"Error loading {video_path}: {str(e)}")
            frames = torch.zeros((self.frame_count, 3, 224, 224))

        return frames, torch.tensor(label, dtype=torch.float32), dataset_name

    def _load_frames_adaptive(self, video_path):
        """
        Adaptive consecutive frame sampling strategy
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            if self.mode == 'train':
                indices = self._adaptive_sample_indices(total_frames)
            else:
                indices = self._uniform_consecutive_indices(total_frames)

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((256, 256, 3), dtype=np.uint8)

                frame = cv2.resize(frame, (256, 256))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                if self.transform:
                    frame = self.transform(frame)
                else:
                    from torchvision import transforms
                    frame = transforms.ToTensor()(frame)
                frames.append(frame)

        finally:
            cap.release()

        if len(frames) < self.frame_count:
            frames.extend([torch.zeros_like(frames[0])] * (self.frame_count - len(frames)))

        return frames[:self.frame_count]

    def _adaptive_sample_indices(self, total_frames):
        T = self.frame_count
        n = self.n_factor

        if total_frames < T:
            return self._cyclic_repetition_indices(total_frames, T)
        elif total_frames <= n * T:
            return self._consecutive_segments_indices(total_frames, T)
        else:
            return self._distributed_segments_indices(total_frames, T)

    def _uniform_consecutive_indices(self, total_frames):
        T = self.frame_count
        if total_frames < T:
            indices = []
            for i in range(T):
                indices.append(i % total_frames)
            return indices
        else:
            start = max(0, (total_frames - T) // 2)
            return list(range(start, min(start + T, total_frames)))

    def _cyclic_repetition_indices(self, total_frames, T):
        indices = []
        for i in range(T):
            idx = i % total_frames
            indices.append(idx)
        if self.mode == 'train':
            offset = random.randint(0, total_frames - 1)
            indices = [(idx + offset) % total_frames for idx in indices]
        return indices

    def _consecutive_segments_indices(self, total_frames, T):
        segment1_size = T // 2
        segment2_size = T - segment1_size

        start1 = 0
        segment1 = list(range(start1, min(start1 + segment1_size, total_frames)))
        start2 = max(0, total_frames - segment2_size)
        segment2 = list(range(start2, total_frames))

        indices = segment1 + segment2

        if self.mode == 'train' and len(indices) == T:
            max_jitter = min(5, total_frames // 10)
            if max_jitter > 0:
                jitter1 = random.randint(-max_jitter, max_jitter)
                jitter2 = random.randint(-max_jitter, max_jitter)
                indices = [max(0, min(total_frames - 1, i + jitter1)) if i < segment1_size
                           else max(0, min(total_frames - 1, i + jitter2))
                           for i in indices]

        return indices

    def _distributed_segments_indices(self, total_frames, T):
        segment_size = T // 3
        remainder = T % 3
        segment_sizes = [segment_size + (1 if i < remainder else 0) for i in range(3)]

        indices = []

        start1 = 0
        end1 = int(total_frames * 0.2)
        segment1 = self._sample_consecutive_frames(start1, end1, segment_sizes[0])
        indices.extend(segment1)

        start2 = int(total_frames * 0.4)
        end2 = int(total_frames * 0.6)
        segment2 = self._sample_consecutive_frames(start2, end2, segment_sizes[1])
        indices.extend(segment2)

        start3 = int(total_frames * 0.8)
        end3 = total_frames
        segment3 = self._sample_consecutive_frames(start3, end3, segment_sizes[2])
        indices.extend(segment3)

        if self.mode == 'train':
            max_jitter = min(3, total_frames // 20)
            if max_jitter > 0:
                indices = [max(0, min(total_frames - 1, i + random.randint(-max_jitter, max_jitter)))
                           for i in indices]

        return indices

    def _sample_consecutive_frames(self, start, end, num_frames):
        if end - start <= num_frames:
            segment = list(range(start, end))
            if len(segment) < num_frames:
                segment = segment * (num_frames // len(segment) + 1)
            return segment[:num_frames]
        else:
            max_start = end - num_frames
            actual_start = random.randint(start, max_start) if self.mode == 'train' else start
            return list(range(actual_start, actual_start + num_frames))


def load_multi_datasets(config, device):
    """
    Load multiple datasets and combine them

    Args:
        config: Configuration dictionary
        device: Device for data loading

    Returns:
        train_loader, val_loader, test_loader, dataset_stats
    """
    all_train_datasets = []
    all_val_datasets = []
    all_test_datasets = []
    dataset_stats = {}

    for dataset_name, dataset_config in config['data']['datasets'].items():
        if not dataset_config.get('enabled', True):
            continue

        print(f"\nLoading dataset: {dataset_name}")

        # Load videos
        fake_videos = get_video_files(dataset_config['fake_path'])
        real_videos = get_video_files(dataset_config['real_path'])

        # Balance the dataset
        min_count = min(len(fake_videos), len(real_videos))
        fake_videos = random.sample(fake_videos, min_count)
        real_videos = random.sample(real_videos, min_count)

        all_videos = real_videos + fake_videos
        all_labels = [1] * len(real_videos) + [0] * len(fake_videos)

        # Split using the specified ratios
        train_ratio, val_ratio, test_ratio = dataset_config['split_ratio']

        # First split: train+val vs test
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            all_videos, all_labels, test_size=test_ratio,
            stratify=all_labels, random_state=config['split']['random_state']
        )

        # Second split: train vs val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels, test_size=val_ratio_adjusted,
            stratify=train_val_labels, random_state=config['split']['random_state']
        )

        # Create datasets
        train_dataset = VideoDataset(
            [p for p, l in zip(train_paths, train_labels) if l == 1],
            [p for p, l in zip(train_paths, train_labels) if l == 0],
            get_transforms('train'),
            frame_count=config['data']['frame_count'],
            mode='train',
            min_frames=config['data']['min_frames'],
            n_factor=config['data']['n_factor'],
            dataset_name=dataset_name
        )

        val_dataset = VideoDataset(
            [p for p, l in zip(val_paths, val_labels) if l == 1],
            [p for p, l in zip(val_paths, val_labels) if l == 0],
            get_transforms('val'),
            frame_count=config['data']['frame_count'],
            mode='val',
            min_frames=config['data']['min_frames'],
            n_factor=config['data']['n_factor'],
            dataset_name=dataset_name
        )

        test_dataset = VideoDataset(
            [p for p, l in zip(test_paths, test_labels) if l == 1],
            [p for p, l in zip(test_paths, test_labels) if l == 0],
            get_transforms('val'),
            frame_count=config['data']['frame_count'],
            mode='test',
            min_frames=config['data']['min_frames'],
            n_factor=config['data']['n_factor'],
            dataset_name=dataset_name
        )

        all_train_datasets.append(train_dataset)
        all_val_datasets.append(val_dataset)
        all_test_datasets.append(test_dataset)

        dataset_stats[dataset_name] = {
            'train': {'real': sum(train_labels), 'fake': len(train_labels) - sum(train_labels)},
            'val': {'real': sum(val_labels), 'fake': len(val_labels) - sum(val_labels)},
            'test': {'real': sum(test_labels), 'fake': len(test_labels) - sum(test_labels)}
        }

    # Combine datasets
    if len(all_train_datasets) > 1:
        train_dataset = ConcatDataset(all_train_datasets)
        val_dataset = ConcatDataset(all_val_datasets)
        test_dataset = ConcatDataset(all_test_datasets)
    else:
        train_dataset = all_train_datasets[0]
        val_dataset = all_val_datasets[0]
        test_dataset = all_test_datasets[0]

    # Create dataloaders with batch size 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    return train_loader, val_loader, test_loader, dataset_stats