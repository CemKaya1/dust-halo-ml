"""
DSH Detector Training - Final Version
======================================
Trains ResNet model with dataset v2 (noise injection + hard negatives).

Features:
- ResNet architecture (skip connections, LeakyReLU, strided conv)
- Dataset v2 (noise injection, hard PSF negatives)
- Output clamping (prevents NaN)
- Gradient clipping (stability)
- Confusion matrix logging
- Full reproducibility (seed everything)

Usage:
    CUDA_VISIBLE_DEVICES="" python3 train_final.py

"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, '.')

from models.dsh_resnet import DSHResNet, count_parameters
from data.dataset_v2 import create_data_loaders_v2


def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed: {seed}")


def train_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0, eps=1e-7):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Clamp outputs to prevent log(0)
        outputs = torch.clamp(outputs, min=eps, max=1.0 - eps)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Confusion matrix
        pred = (outputs > 0.5).float().cpu().numpy().flatten()
        true = labels.cpu().numpy().flatten()
        
        tp += np.sum((pred == 1) & (true == 1))
        tn += np.sum((pred == 0) & (true == 0))
        fp += np.sum((pred == 1) & (true == 0))
        fn += np.sum((pred == 0) & (true == 1))
        
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch [{batch_idx + 1}/{len(loader)}] Loss: {loss.item():.4f}")
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return total_loss / len(loader), accuracy, {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


@torch.no_grad()
def validate(model, loader, criterion, device, eps=1e-7):
    """Validate the model."""
    model.eval()
    total_loss = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    all_probs = []
    
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        outputs_clamped = torch.clamp(outputs, min=eps, max=1.0 - eps)
        
        loss = criterion(outputs_clamped, labels)
        total_loss += loss.item()
        
        pred = (outputs > 0.5).float().cpu().numpy().flatten()
        true = labels.cpu().numpy().flatten()
        
        tp += np.sum((pred == 1) & (true == 1))
        tn += np.sum((pred == 0) & (true == 0))
        fp += np.sum((pred == 1) & (true == 0))
        fn += np.sum((pred == 0) & (true == 1))
        
        all_probs.extend(outputs.cpu().numpy().flatten())
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
        'prob_mean': float(np.mean(all_probs)),
        'prob_std': float(np.std(all_probs))
    }


def print_confusion(cm, title=""):
    """Print confusion matrix."""
    print(f"\n  {title}Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 No Halo  Halo")
    print(f"  Actual No Halo  {cm['tn']:5d}  {cm['fp']:5d}")
    print(f"  Actual Halo     {cm['fn']:5d}  {cm['tp']:5d}")


def main():
    print("=" * 60)
    print("DSH DETECTOR - FINAL TRAINING")
    print("=" * 60)
    
    # ===== CONFIGURATION =====
    SEED = 42
    CSV_PATH = 'balanced_10k_vND7_with_split.csv'
    DATA_ROOT = '/data3/efeoztaban/vND7_directories_shrunk_clouds/'
    EROSITA_BG = '/data3/pure26/eRASS/test_data/150/203/EXP_010/em01_203150_024_Image_c010.fits.gz'
    
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    PATIENCE = 10
    
    CHECKPOINT_DIR = './checkpoints_final'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Seed
    seed_everything(SEED)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data
    print("\nLoading data (v2 with noise injection + hard negatives)...")
    train_loader, val_loader, test_loader = create_data_loaders_v2(
        csv_path=CSV_PATH,
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=4,
        negative_ratio=1.0,
        erosita_background_path=EROSITA_BG,
        add_noise_to_positives=True,
        noise_scale=0.1
    )
    
    # Model
    print("\nCreating ResNet model...")
    model = DSHResNet(dropout_rate=0.3)
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    
    # History
    history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print("-" * 40)
        
        # Train
        train_loss, train_acc, train_cm = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val = validate(model, val_loader, criterion, device)
        
        # Scheduler
        scheduler.step(val['loss'])
        lr = optimizer.param_groups[0]['lr']
        
        # Log
        history['train'].append({'loss': train_loss, 'acc': train_acc, 'cm': train_cm})
        history['val'].append(val)
        
        # Print
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val['loss']:.4f} | Acc: {val['accuracy']:.4f}")
        print(f"  Precision: {val['precision']:.4f} | Recall: {val['recall']:.4f} | F1: {val['f1']:.4f}")
        print(f"  Prob: mean={val['prob_mean']:.4f}, std={val['prob_std']:.4f}")
        print(f"  LR: {lr:.2e}")
        
        print_confusion(val['confusion'], "Val ")
        
        # Save best
        if val['loss'] < best_val_loss:
            best_val_loss = val['loss']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val,
                'config': {
                    'seed': SEED,
                    'model': 'DSHResNet',
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE
                }
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            
            print("  ✓ Best model saved!")
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    total_time = time.time() - start_time
    
    # Save history
    with open(os.path.join(CHECKPOINT_DIR, 'history.json'), 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        json.dump(convert(history), f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Time: {total_time / 60:.1f} minutes")
    print(f"Best val loss: {best_val_loss:.4f}")
    
    # Test
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    
    checkpoint = torch.load(
        os.path.join(CHECKPOINT_DIR, 'best_model.pth'),
        map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test = validate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test['loss']:.4f}")
    print(f"Test Accuracy: {test['accuracy']:.4f}")
    print(f"Test Precision: {test['precision']:.4f}")
    print(f"Test Recall: {test['recall']:.4f}")
    print(f"Test F1: {test['f1']:.4f}")
    print_confusion(test['confusion'], "Test ")
    
    print("\n" + "=" * 60)
    print(f"Model saved: {CHECKPOINT_DIR}/best_model.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()