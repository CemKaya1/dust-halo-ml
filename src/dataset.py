# src/data/halo_dataset.py

import os
from typing import Optional, Callable, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from astropy.io import fits


class HaloDataset(Dataset):
    """
    Dataset for synthetic dust-scattering halo images.

    Expects a CSV with at least:
      - 'relative_path' or 'path': path to FITS image (relative to images_root)
      - 'split': e.g. 'train' / 'validation'
      - 'dist_int' or 'distance': source distance (for binning)
      - 'nh_unif_idx' or similar: NH class index

    Returns:
      - image: torch.FloatTensor of shape (1, H, W)
      - labels: dict with tensors (e.g. distance_bin, nh_class)
    """

    def __init__(
        self,
        csv_path: str,
        images_root: str,
        split: Optional[str] = None,
        split_column: str = "split",
        image_column: str = "relative_path",
        distance_column: str = "dist_int",
        nh_column: str = "nh_unif_idx",
        transform: Optional[Callable] = None,
        distance_bin_edges: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        self.csv_path = csv_path
        self.images_root = images_root
        self.image_column = image_column
        self.distance_column = distance_column
        self.nh_column = nh_column
        self.transform = transform

        df = pd.read_csv(csv_path)

        # Filter by split if requested
        if split is not None and split_column in df.columns:
            df = df[df[split_column] == split].copy()

        df = df.reset_index(drop=True)
        self.df = df

        # Distance binning
        distances = df[distance_column].values.astype(float)

        if distance_bin_edges is None:
            percentiles = [0, 25, 50, 75, 100]
            self.distance_bin_edges = np.percentile(distances, percentiles)
            self.distance_bin_edges[0] -= 1e-6
            self.distance_bin_edges[-1] += 1e-6
        else:
            self.distance_bin_edges = np.array(distance_bin_edges, dtype=float)

        self.distance_bins = np.digitize(
            distances, self.distance_bin_edges[1:-1], right=False
        )  

    def __len__(self) -> int:
        return len(self.df)

    def _load_fits_image(self, rel_path: str) -> np.ndarray:
        full_path = os.path.join(self.images_root, rel_path)
        with fits.open(full_path, memmap=True) as hdul:
            image = hdul[0].data.astype(np.float32)

        # Basic NaN/inf handling
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # Basic normalization: subtract median, divide by 99th percentile
        median = np.median(image)
        image = image - median
        p99 = np.percentile(np.abs(image), 99)
        if p99 > 0:
            image = image / p99

        return image

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        rel_path = row[self.image_column]
        image = self._load_fits_image(rel_path)

        # Convert to torch tensor (1, H, W)
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # add channel dimension

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        # Labels
        distance_bin = self.distance_bins[idx]
        nh_class = int(row[self.nh_column])

        labels = {
            "distance_bin": torch.tensor(distance_bin, dtype=torch.long),
            "nh_class": torch.tensor(nh_class, dtype=torch.long),
        }

        return {
            "image": image_tensor,
            "labels": labels,
        }
