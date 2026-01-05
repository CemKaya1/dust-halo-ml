"""
DSH Dataset v2 - With Critical Improvements
============================================
Improved dataset addressing the domain gap problem:

1. Noise Injection to Positives
   - Adds Poisson noise to synthetic DSH images
   - Prevents model from learning "smooth = halo, grainy = no halo"
   
2. Hard Negatives (Bright Point Sources)
   - Generates stars WITHOUT halo rings
   - Forces model to learn ring structure, not just "bright center = halo"
   
3. Metadata Parity
   - All samples (positive and negative) have identical dictionary keys
   - Prevents DataLoader crashes during batching

Author: DSH Detection Project - Part 5 (Model Selection & Training)
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits


class DSHDatasetV2(Dataset):
    """
    Improved Dataset with noise injection and hard negatives.
    
    Key improvements:
    1. Positives get Poisson noise added (match negative graininess)
    2. Hard negatives: bright PSF sources without halos
    3. All metadata dictionaries have identical keys
    
    Args:
        csv_path: Path to CSV with positive sample metadata
        data_root: Root directory for FITS files
        split: Dataset split ('train', 'val', 'test')
        image_size: Target image size
        negative_ratio: Ratio of negative to positive samples
        erosita_background_path: Optional path to eROSITA for background extraction
        add_noise_to_positives: Whether to add Poisson noise to positives
        noise_scale: Scale factor for noise (higher = more noise)
    """
    
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        split: str = 'train',
        image_size: int = 64,
        negative_ratio: float = 1.0,
        erosita_background_path: Optional[str] = None,
        add_noise_to_positives: bool = True,
        noise_scale: float = 0.1,
        augment: bool = True
    ):
        self.data_root = data_root
        self.image_size = image_size
        self.negative_ratio = negative_ratio
        self.add_noise_to_positives = add_noise_to_positives
        self.noise_scale = noise_scale
        self.augment = augment and (split == 'train')
        
        # Load positive samples metadata
        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.n_positives = len(self.df)
        
        # Get column names for metadata parity
        self.metadata_columns = list(df.columns)
        
        # Calculate negatives
        self.n_negatives = int(self.n_positives * negative_ratio)
        self.total_samples = self.n_positives + self.n_negatives
        
        # Load eROSITA backgrounds
        self.erosita_backgrounds = None
        if erosita_background_path and os.path.exists(erosita_background_path):
            self._load_erosita_backgrounds(erosita_background_path)
        
        print(f"[{split.upper()}] Positives: {self.n_positives}, Negatives: {self.n_negatives}")
        print(f"[{split.upper()}] Total: {self.total_samples}")
        print(f"[{split.upper()}] Noise injection: {add_noise_to_positives}, scale={noise_scale}")
        if self.erosita_backgrounds:
            print(f"[{split.upper()}] Loaded {len(self.erosita_backgrounds)} eROSITA backgrounds")
    
    def _load_erosita_backgrounds(self, path: str, n_patches: int = 500):
        """Extract background patches from eROSITA."""
        try:
            with fits.open(path) as hdul:
                image = hdul[0].data.astype(np.float32)
            
            h, w = image.shape
            patches = []
            
            for _ in range(n_patches):
                x = np.random.randint(0, w - self.image_size)
                y = np.random.randint(0, h - self.image_size)
                patch = image[y:y+self.image_size, x:x+self.image_size].copy()
                patches.append(patch)
            
            self.erosita_backgrounds = patches
            
        except Exception as e:
            print(f"Warning: Could not load eROSITA backgrounds: {e}")
            self.erosita_backgrounds = None
    
    def __len__(self) -> int:
        return self.total_samples
    
    def _load_fits_image(self, path: str) -> np.ndarray:
        """Load FITS file."""
        try:
            with fits.open(path) as hdul:
                image = hdul[0].data
                if image is None and len(hdul) > 1:
                    image = hdul[1].data
                return image.astype(np.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        from scipy.ndimage import zoom
        
        if image.shape[0] == self.image_size and image.shape[1] == self.image_size:
            return image
        
        zh = self.image_size / image.shape[0]
        zw = self.image_size / image.shape[1]
        return zoom(image, (zh, zw), order=1)
    
    def _add_poisson_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add Poisson noise to make synthetic images look grainy like real X-ray.
        
        This is CRITICAL - prevents model from learning "smooth = halo".
        """
        # Normalize to reasonable count range
        img_max = image.max()
        if img_max > 0:
            # Scale to counts (higher scale = more visible noise)
            scale = self.noise_scale / img_max
            counts = image * scale
            
            # Add Poisson noise
            noisy = np.random.poisson(np.clip(counts, 0, None) + 0.001)
            
            # Scale back
            noisy = noisy.astype(np.float32) / scale
        else:
            noisy = image
        
        return noisy
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1]."""
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Log transform
        image = np.log1p(np.clip(image, 0, None))
        
        # Min-max normalize
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image)
        
        return image
    
    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        if not self.augment:
            return image
        
        if np.random.random() > 0.5:
            image = np.fliplr(image).copy()
        if np.random.random() > 0.5:
            image = np.flipud(image).copy()
        
        k = np.random.randint(0, 4)
        image = np.rot90(image, k).copy()
        
        return image
    
    def _get_dummy_metadata(self, negative_type: str, idx: int) -> Dict:
        """
        Create metadata with same keys as positive samples.
        
        CRITICAL: All samples must have identical dictionary keys
        or DataLoader will crash during batching.
        """
        metadata = {
            'filename': f'negative_{negative_type}_{idx}',
            'type': 'negative',
            'negative_type': negative_type,
            'is_positive': False,
            'label': 0,
            # Match all columns from the CSV
            'distance': -1.0,
            'nh_unif': 'NONE',
            'nh_wco': 'NONE',
            'cloud_bits': '0' * 15,
            'energy_band': 'NONE',
            'split': 'generated',
            'relative_path': 'generated',
            'path': 'generated'
        }
        
        # Add any other columns from the original CSV
        for col in self.metadata_columns:
            if col not in metadata:
                metadata[col] = 'NONE' if isinstance(self.df[col].iloc[0], str) else -1
        
        return metadata
    
    def _get_positive_metadata(self, row: pd.Series) -> Dict:
        """Create metadata for positive sample with all required keys."""
        metadata = {
            'filename': row.get('filename', 'unknown'),
            'type': 'positive',
            'negative_type': 'NONE',
            'is_positive': True,
            'label': 1,
            'distance': float(row.get('distance', -1)),
            'nh_unif': str(row.get('nh_unif', 'NONE')),
            'nh_wco': str(row.get('nh_wco', 'NONE')),
            'cloud_bits': str(row.get('cloud_bits', '0' * 15)),
            'energy_band': str(row.get('energy_band', 'NONE')),
            'split': str(row.get('split', 'unknown')),
            'relative_path': str(row.get('relative_path', '')),
            'path': str(row.get('path', ''))
        }
        
        # Add any other columns
        for col in self.metadata_columns:
            if col not in metadata:
                val = row.get(col, None)
                metadata[col] = str(val) if isinstance(val, str) else float(val) if val is not None else -1
        
        return metadata
    
    # ==================== NEGATIVE SAMPLE GENERATORS ====================
    
    def _generate_hard_negative_psf(self) -> np.ndarray:
        """
        Generate HARD negative: Bright point source WITHOUT halo.
        
        This is the critical test - can the model tell the difference between
        a star WITH a halo ring vs a star WITHOUT a halo ring?
        
        The PSF (Point Spread Function) is just a Gaussian - no ring structure.
        """
        image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Create coordinate grid
        x = np.arange(self.image_size)
        y = np.arange(self.image_size)
        xx, yy = np.meshgrid(x, y)
        
        # Random center (allow some offset from center)
        cx = np.random.randint(self.image_size // 4, 3 * self.image_size // 4)
        cy = np.random.randint(self.image_size // 4, 3 * self.image_size // 4)
        
        # PSF parameters - make it realistic
        # Bright central peak
        peak_intensity = np.random.uniform(10, 100)
        sigma_core = np.random.uniform(1.5, 3.0)  # Tight core
        
        # Main PSF (Gaussian)
        psf = peak_intensity * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma_core**2))
        
        # Optional: Add PSF wings (broader, fainter)
        if np.random.random() > 0.5:
            wing_intensity = peak_intensity * np.random.uniform(0.01, 0.05)
            sigma_wing = sigma_core * np.random.uniform(2, 4)
            wings = wing_intensity * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma_wing**2))
            psf += wings
        
        image = psf
        
        # Add Poisson noise (realistic X-ray)
        image = np.random.poisson(np.clip(image, 0, None) + 0.001).astype(np.float32)
        
        return image
    
    def _generate_multiple_psf(self) -> np.ndarray:
        """Generate multiple point sources (star field) without halos."""
        image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        x = np.arange(self.image_size)
        y = np.arange(self.image_size)
        xx, yy = np.meshgrid(x, y)
        
        # Random number of sources
        n_sources = np.random.randint(2, 6)
        
        for _ in range(n_sources):
            cx = np.random.randint(5, self.image_size - 5)
            cy = np.random.randint(5, self.image_size - 5)
            
            intensity = np.random.uniform(5, 50)
            sigma = np.random.uniform(1.5, 3.5)
            
            psf = intensity * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            image += psf
        
        # Add Poisson noise
        image = np.random.poisson(np.clip(image, 0, None) + 0.001).astype(np.float32)
        
        return image
    
    def _generate_poisson_background(self) -> np.ndarray:
        """Generate sparse Poisson background (no sources)."""
        mean_counts = np.random.uniform(0.001, 0.02)
        image = np.random.poisson(mean_counts, (self.image_size, self.image_size))
        return image.astype(np.float32)
    
    def _generate_gradient_background(self) -> np.ndarray:
        """Generate smooth gradient background."""
        x = np.linspace(0, 1, self.image_size)
        y = np.linspace(0, 1, self.image_size)
        xx, yy = np.meshgrid(x, y)
        
        angle = np.random.uniform(0, 2 * np.pi)
        gradient = np.cos(angle) * xx + np.sin(angle) * yy
        gradient = gradient * np.random.uniform(0.5, 2.0)
        
        # Add noise
        gradient = np.random.poisson(np.clip(gradient, 0, None) + 0.001)
        return gradient.astype(np.float32)
    
    def _generate_random_structures(self) -> np.ndarray:
        """Generate random non-ring structures."""
        image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Random bright pixels
        n_pixels = np.random.randint(5, 30)
        for _ in range(n_pixels):
            x = np.random.randint(0, self.image_size)
            y = np.random.randint(0, self.image_size)
            image[y, x] = np.random.uniform(1, 20)
        
        # Maybe add streak
        if np.random.random() > 0.7:
            length = np.random.randint(5, 15)
            start_x = np.random.randint(0, self.image_size - length)
            start_y = np.random.randint(0, self.image_size)
            angle = np.random.uniform(0, np.pi)
            
            for i in range(length):
                x = int(start_x + i * np.cos(angle))
                y = int(start_y + i * np.sin(angle))
                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    image[y, x] = np.random.uniform(3, 10)
        
        image += np.random.poisson(0.002, image.shape)
        return image.astype(np.float32)
    
    def _get_erosita_background(self) -> np.ndarray:
        """Get random real eROSITA background patch."""
        if self.erosita_backgrounds is None or len(self.erosita_backgrounds) == 0:
            return self._generate_poisson_background()
        
        idx = np.random.randint(0, len(self.erosita_backgrounds))
        return self.erosita_backgrounds[idx].copy()
    
    def _generate_negative(self, idx: int) -> Tuple[np.ndarray, str]:
        """
        Generate a negative sample.
        
        Distribution:
        - 30% Hard negatives (bright PSF without halo) - CRITICAL
        - 20% Multiple PSF sources
        - 15% Poisson background
        - 15% Real eROSITA backgrounds  
        - 10% Gradient backgrounds
        - 10% Random structures
        """
        np.random.seed(idx * 7919)  # Reproducibility
        
        choice = np.random.random()
        
        if choice < 0.30:
            # HARD NEGATIVES - Most important!
            image = self._generate_hard_negative_psf()
            neg_type = 'hard_psf'
        elif choice < 0.50:
            image = self._generate_multiple_psf()
            neg_type = 'multiple_psf'
        elif choice < 0.65:
            image = self._generate_poisson_background()
            neg_type = 'poisson'
        elif choice < 0.80:
            image = self._get_erosita_background()
            neg_type = 'erosita_bg'
        elif choice < 0.90:
            image = self._generate_gradient_background()
            neg_type = 'gradient'
        else:
            image = self._generate_random_structures()
            neg_type = 'random'
        
        return image, neg_type
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get a sample."""
        
        if idx < self.n_positives:
            # ========== POSITIVE SAMPLE (DSH) ==========
            row = self.df.iloc[idx]
            fits_path = os.path.join(self.data_root, row['relative_path'])
            
            image = self._load_fits_image(fits_path)
            image = self._resize_image(image)
            
            # CRITICAL: Add noise to positives!
            # This prevents model from learning "smooth = halo"
            if self.add_noise_to_positives:
                image = self._add_poisson_noise(image)
            
            image = self._normalize(image)
            image = self._augment(image)
            
            label = 1.0
            metadata = self._get_positive_metadata(row)
            
        else:
            # ========== NEGATIVE SAMPLE ==========
            neg_idx = idx - self.n_positives
            image, neg_type = self._generate_negative(neg_idx)
            
            image = self._normalize(image)
            image = self._augment(image)
            
            label = 0.0
            metadata = self._get_dummy_metadata(neg_type, neg_idx)
        
        # Convert to tensor
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.tensor([label], dtype=torch.float32)
        
        return image, label, metadata


def create_data_loaders_v2(
    csv_path: str,
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 64,
    negative_ratio: float = 1.0,
    erosita_background_path: Optional[str] = None,
    add_noise_to_positives: bool = True,
    noise_scale: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test data loaders with v2 dataset."""
    
    train_dataset = DSHDatasetV2(
        csv_path=csv_path,
        data_root=data_root,
        split='train',
        image_size=image_size,
        negative_ratio=negative_ratio,
        erosita_background_path=erosita_background_path,
        add_noise_to_positives=add_noise_to_positives,
        noise_scale=noise_scale,
        augment=True
    )
    
    val_dataset = DSHDatasetV2(
        csv_path=csv_path,
        data_root=data_root,
        split='val',
        image_size=image_size,
        negative_ratio=negative_ratio,
        erosita_background_path=erosita_background_path,
        add_noise_to_positives=add_noise_to_positives,
        noise_scale=noise_scale,
        augment=False
    )
    
    test_dataset = DSHDatasetV2(
        csv_path=csv_path,
        data_root=data_root,
        split='test',
        image_size=image_size,
        negative_ratio=negative_ratio,
        erosita_background_path=erosita_background_path,
        add_noise_to_positives=add_noise_to_positives,
        noise_scale=noise_scale,
        augment=False
    )
    
    # Custom collate to handle metadata dicts
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        # Keep metadata as list of dicts
        metadata = [item[2] for item in batch]
        return images, labels, metadata
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("DSH Dataset v2 - Improved")
    print("=" * 60)
    print("""
Key Improvements:
1. Noise injection to positives (prevents "smooth = halo" learning)
2. Hard negatives: Bright PSF without halo rings (30% of negatives)
3. Metadata parity: All samples have identical dictionary keys

Negative Distribution:
- 30% Hard PSF (bright star, no halo) - CRITICAL
- 20% Multiple PSF sources
- 15% Poisson background
- 15% Real eROSITA backgrounds
- 10% Gradient backgrounds
- 10% Random structures

Usage:
    from dataset_v2 import create_data_loaders_v2
    
    train_loader, val_loader, test_loader = create_data_loaders_v2(
        csv_path='balanced_10k_vND7_with_split.csv',
        data_root='/data3/efeoztaban/vND7_directories_shrunk_clouds/',
        erosita_background_path='/path/to/erosita.fits',
        add_noise_to_positives=True,
        noise_scale=0.1
    )
    """)