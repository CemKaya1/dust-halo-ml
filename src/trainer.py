# src/training/train_baseline.py

import argparse
import os
from typing import Dict

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.halo_dataset import HaloDataset
from src.models.custom_cnn import HaloCNN


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    device = torch.device(cfg["training"]["device"])

    # --- Dataset & Dataloaders ---
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]
    train_dataset = HaloDataset(
        csv_path=paths_cfg["csv_path"],
        images_root=paths_cfg["images_root"],
        split=cfg["data"]["train_split_value"],
        split_column="split",
        image_column=data_cfg.get("image_key", "relative_path"),
        distance_column=data_cfg.get("distance_column", "dist_int"),
        nh_column=data_cfg.get("nh_unif_column", "nh_unif_idx"),
        transform=None,  # TODO: plug in augmentations
    )

    val_dataset = HaloDataset(
        csv_path=paths_cfg["csv_path"],
        images_root=paths_cfg["images_root"],
        split=cfg["data"]["val_split_value"],
        split_column="split",
        image_column=data_cfg.get("image_key", "relative_path"),
        distance_column=data_cfg.get("distance_column", "dist_int"),
        nh_column=data_cfg.get("nh_unif_column", "nh_unif_idx"),
        transform=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    # --- Model ---
    model = HaloCNN(
        num_distance_bins=data_cfg["num_distance_bins"],
        num_nh_classes=data_cfg["num_nh_classes"],
        include_auxiliary=True,
    ).to(device)

    # --- Losses ---
    bce_loss = nn.BCEWithLogitsLoss()     # for halo vs non-halo
    ce_loss = nn.CrossEntropyLoss()       # for distance/NH

    lambda_halo = 0.0
    lambda_dist = 1.0
    lambda_nh = 1.0

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    num_epochs = cfg["training"]["num_epochs"]

    for epoch in range(1, num_epochs + 1):
        # ---------- TRAIN ----------
        model.train()
        train_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]")

        for batch in pbar:
            images = batch["image"].to(device)                  # (B, 1, H, W)
            labels = batch["labels"]
            distance_bins = labels["distance_bin"].to(device)    # (B,)
            nh_classes = labels["nh_class"].to(device)           # (B,)

            # For now, assume all are halo-positive (label = 1)
            # Later, replace with real halo/non-halo labels.
            halo_labels = torch.ones(images.size(0), 1, device=device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_halo = bce_loss(outputs["halo_logits"], halo_labels)
            loss_dist = ce_loss(outputs["distance_logits"], distance_bins)
            loss_nh = ce_loss(outputs["nh_logits"], nh_classes)

            loss = lambda_halo * loss_halo + lambda_dist * loss_dist + lambda_nh * loss_nh
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": train_loss / num_batches})

        avg_train_loss = train_loss / max(num_batches, 1)

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.inference_mode():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["labels"]
                distance_bins = labels["distance_bin"].to(device)
                nh_classes = labels["nh_class"].to(device)
                halo_labels = torch.ones(images.size(0), 1, device=device)

                outputs = model(images)

                loss_halo = bce_loss(outputs["halo_logits"], halo_labels)
                loss_dist = ce_loss(outputs["distance_logits"], distance_bins)
                loss_nh = ce_loss(outputs["nh_logits"], nh_classes)

                loss = lambda_halo * loss_halo + lambda_dist * loss_dist + lambda_nh * loss_nh

                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / max(num_val_batches, 1)
        print(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}"
        )

        # TODO: add metric calculations per-bin, save best model, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    main(args.config)
