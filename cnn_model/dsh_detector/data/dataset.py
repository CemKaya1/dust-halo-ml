"""
DSH Dataset Module
==================
Custom PyTorch Dataset for loading dust scattering halo images from FITS files.
Includes support for generating negative samples (black images).

"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits


class DSHDataset(Dataset):
    """
    PyTorch Dataset for Dust Scattering Halo detection.
    
    Loads FITS images from the synthetic dataset and optionally generates
    black images as negative samples for binary classification.
    
    Args:
        csv_path: Path to the CSV file containing image metadata
        data_root: Root directory where FITS files are stored
        split: Dataset split ('train', 'val', 'test')
        image_size: Target image size (will resize if different)
        include_negatives: Whether to include black images as negatives
        negative_ratio: Ratio of negative to positive samples (default 1.0 = equal)
        transform: Optional custom transforms to apply
        normalize: Whether to normalize images to [0, 1]
        log_transform: Whether to apply log transformation (good for X-ray data)
    """
    
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        split: str = 'train',
        image_size: int = 64,
        include_negatives: bool = True,
        negative_ratio: float = 1.0,
        transform: Optional[Any] = None,
        normalize: bool = True,
        log_transform: bool = True
    ):
        self.data_root = data_root
        self.image_size = image_size
        self.include_negatives = include_negatives
        self.negative_ratio = negative_ratio
        self.transform = transform
        self.normalize = normalize
        self.log_transform = log_transform
        
        # Load metadata
        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        
        print(f"[{split.upper()}] Loaded {len(self.df)} positive samples")
        
        # Calculate number of negative samples
        if include_negatives:
            self.n_negatives = int(len(self.df) * negative_ratio)
            print(f"[{split.upper()}] Will generate {self.n_negatives} negative samples")
        else:
            self.n_negatives = 0
        
        self.total_samples = len(self.df) + self.n_negatives
        print(f"[{split.upper()}] Total samples: {self.total_samples}")
    
    def __len__(self) -> int:
        return self.total_samples
    
    def _load_fits_image(self, path: str) -> np.ndarray:
        """Load a FITS file and return the image data."""
        try:
            with fits.open(path) as hdul:
                # Usually the image is in the primary HDU or first extension
                image = hdul[0].data
                if image is None and len(hdul) > 1:
                    image = hdul[1].data
                return image.astype(np.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a zero image on error
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image (resize, normalize, log transform)."""
        # Handle NaN and Inf values
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Resize if necessary
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = self._resize_image(image)
        
        # Log transform (useful for X-ray images with large dynamic range)
        if self.log_transform:
            # Add small epsilon to avoid log(0)
            image = np.log1p(np.clip(image, 0, None))
        
        # Normalize to [0, 1]
        if self.normalize:
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
            else:
                image = np.zeros_like(image)
        
        return image
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size using bilinear interpolation."""
        from scipy.ndimage import zoom
        
        h, w = image.shape
        zoom_h = self.image_size / h
        zoom_w = self.image_size / w
        return zoom(image, (zoom_h, zoom_w), order=1)
    
    def _generate_black_image(self) -> np.ndarray:
        """Generate a black (zero) image as a negative sample."""
        return np.zeros((self.image_size, self.image_size), dtype=np.float32)
    
    def _generate_noise_image(self, noise_level: float = 0.05) -> np.ndarray:
        """Generate a noise-only image as a negative sample."""
        noise = np.random.randn(self.image_size, self.image_size).astype(np.float32)
        noise = noise * noise_level
        noise = np.clip(noise, 0, 1)
        return noise
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (image, label, metadata)
            - image: Tensor of shape (1, H, W)
            - label: Tensor of shape (1,) with 0 or 1
            - metadata: Dictionary with sample information
        """
        if idx < len(self.df):
            # Positive sample (DSH image)
            row = self.df.iloc[idx]
            
            # Construct full path
            fits_path = os.path.join(self.data_root, row['relative_path'])
            
            # Load and preprocess image
            image = self._load_fits_image(fits_path)
            image = self._preprocess_image(image)
            
            label = 1.0
            metadata = {
                'filename': row['filename'],
                'distance': row['distance'],
                'nh_unif': row['nh_unif'],
                'nh_wco': row['nh_wco'],
                'cloud_bits': str(row['cloud_bits']),
                'is_synthetic_negative': False
            }
        else:
            # Negative sample (black or noise image)
            # Use deterministic seed based on index for reproducibility
            np.random.seed(idx)
            
            # 80% black images, 20% noise images
            if np.random.random() < 0.8:
                image = self._generate_black_image()
            else:
                image = self._generate_noise_image()
            
            label = 0.0
            metadata = {
                'filename': f'synthetic_negative_{idx}',
                'distance': -1,
                'nh_unif': 'NONE',
                'nh_wco': 'NONE',
                'cloud_bits': '0' * 15,
                'is_synthetic_negative': True
            }
        
        # Apply custom transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor and add channel dimension
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        label = torch.tensor([label], dtype=torch.float32)  # (1,)
        
        return image, label, metadata


class DSHDatasetWithAugmentation(DSHDataset):
    """
    Extended DSH Dataset with data augmentation for training.
    
    Augmentations included:
    - Random rotation (small angles to preserve halo structure)
    - Random flip (horizontal and vertical)
    - Random noise injection
    - Random brightness adjustment
    """
    
    def __init__(self, *args, augment: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations to the image."""
        if not self.augment:
            return image
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.fliplr(image).copy()
        
        # Random vertical flip
        if np.random.random() > 0.5:
            image = np.flipud(image).copy()
        
        # Random 90-degree rotations
        k = np.random.randint(0, 4)
        image = np.rot90(image, k).copy()
        
        # Random noise injection (small amount)
        if np.random.random() > 0.7:
            noise = np.random.randn(*image.shape).astype(np.float32) * 0.02
            image = np.clip(image + noise, 0, 1)
        
        # Random brightness adjustment
        if np.random.random() > 0.7:
            brightness = np.random.uniform(0.9, 1.1)
            image = np.clip(image * brightness, 0, 1)
        
        return image
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        image, label, metadata = super().__getitem__(idx)
        
        # Apply augmentation only to training images (not negatives)
        if self.augment and not metadata['is_synthetic_negative']:
            image_np = image.squeeze(0).numpy()
            image_np = self._augment_image(image_np)
            image = torch.from_numpy(image_np).unsqueeze(0)
        
        return image, label, metadata


def create_data_loaders(
    csv_path: str,
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 64,
    include_negatives: bool = True,
    negative_ratio: float = 1.0,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        csv_path: Path to the CSV file with image metadata
        data_root: Root directory containing FITS files
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        image_size: Target image size
        include_negatives: Whether to include synthetic negative samples
        negative_ratio: Ratio of negatives to positives
        augment_train: Whether to apply augmentation during training
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Training dataset with augmentation
    if augment_train:
        train_dataset = DSHDatasetWithAugmentation(
            csv_path=csv_path,
            data_root=data_root,
            split='train',
            image_size=image_size,
            include_negatives=include_negatives,
            negative_ratio=negative_ratio,
            augment=True
        )
    else:
        train_dataset = DSHDataset(
            csv_path=csv_path,
            data_root=data_root,
            split='train',
            image_size=image_size,
            include_negatives=include_negatives,
            negative_ratio=negative_ratio
        )
    
    # Validation dataset (no augmentation)
    val_dataset = DSHDataset(
        csv_path=csv_path,
        data_root=data_root,
        split='val',
        image_size=image_size,
        include_negatives=include_negatives,
        negative_ratio=negative_ratio
    )
    
    # Test dataset (no augmentation)
    test_dataset = DSHDataset(
        csv_path=csv_path,
        data_root=data_root,
        split='test',
        image_size=image_size,
        include_negatives=include_negatives,
        negative_ratio=negative_ratio
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset (without actual FITS files)
    print("=" * 60)
    print("DSH Dataset Module - Test")
    print("=" * 60)
    
    # This would be run on the server with actual data
    print("\nUsage example:")
    print("""
    from dataset import create_data_loaders
    
    train_loader, val_loader, test_loader = create_data_loaders(
        csv_path='balanced_10k_vND7_with_split.csv',
        data_root='/data3/efeoztaban/vND7_directories_shrunk_clouds/',
        batch_size=32,
        num_workers=4,
        include_negatives=True,
        negative_ratio=1.0
    )
    
    for images, labels, metadata in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Labels: {labels.squeeze().tolist()[:10]}")
        break
    """)
