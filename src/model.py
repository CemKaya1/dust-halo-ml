# src/models/custom_cnn.py

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic Conv -> BatchNorm -> ReLU -> MaxPool block.

    You can tweak kernel size, padding, and pooling later.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        if self.pool is not None:
            x = self.pool(x)
        return x


class HaloCNN(nn.Module):
    """
    Custom CNN for dust-scattering halo images.

    - Input:  image tensor of shape (B, 1, H, W)
    - Outputs:
        "halo_logits":    (B, 1)  - binary halo/non-halo (logit)
        "distance_logits":(B, num_distance_bins)
        "nh_logits":      (B, num_nh_classes)

    """

    def __init__(
        self,
        num_distance_bins: int,
        num_nh_classes: int,
        include_auxiliary: bool = True,
    ) -> None:
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            ConvBlock(1, 32),     # (B, 32, H/2,   W/2)
            ConvBlock(32, 64),    # (B, 64, H/4,   W/4)
            ConvBlock(64, 128),   # (B, 128, H/8,  W/8)
            ConvBlock(128, 256),  # (B, 256, H/16, W/16)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> (B, 256, 1, 1)
        self.fc_shared = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.3)

        # Halo detection head (binary classification)
        self.halo_head = nn.Linear(128, 1)

        # Auxiliary heads
        self.include_auxiliary = include_auxiliary
        if include_auxiliary:
            self.distance_head = nn.Linear(128, num_distance_bins)
            self.nh_head = nn.Linear(128, num_nh_classes)
        else:
            self.distance_head = None
            self.nh_head = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, 1, H, W)

        Returns:
            dict with logits for each head.
        """
        # Feature extraction
        x = self.features(x)               # (B, C, H', W')
        x = self.global_pool(x)            # (B, C, 1, 1)
        x = x.view(x.size(0), -1)          # (B, C)
        x = self.fc_shared(x)              # (B, 128)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)

        outputs: Dict[str, torch.Tensor] = {}

        halo_logits = self.halo_head(x)    # (B, 1)
        outputs["halo_logits"] = halo_logits

        if self.include_auxiliary:
            distance_logits = self.distance_head(x)  # (B, num_distance_bins)
            nh_logits = self.nh_head(x)              # (B, num_nh_classes)
            outputs["distance_logits"] = distance_logits
            outputs["nh_logits"] = nh_logits

        return outputs
