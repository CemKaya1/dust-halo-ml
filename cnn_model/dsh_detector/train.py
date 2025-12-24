"""
DSH Detector Training Script
============================
Complete training pipeline for the Dust Scattering Halo detection CNN.

Features:
- Training with validation monitoring
- Early stopping
- Learning rate scheduling
- Model checkpointing
- TensorBoard logging
- Mixed precision training (optional)


Usage:
    python train.py --config config.yaml
    
    OR with command line arguments:
    python train.py --csv_path data.csv --data_root /path/to/fits/ --epochs 100
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dsh_cnn import DSHDetectorCNN, DSHDetectorCNNLite, count_parameters
from data.dataset import create_data_loaders


class Trainer:
    """
    Trainer class for DSH detection model.
    
    Handles the complete training workflow including:
    - Training and validation loops
    - Metric tracking
    - Model checkpointing
    - Early stopping
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss function (Binary Cross Entropy - already have sigmoid in model)
        self.criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # History for plotting
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Create checkpoint directory
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', False) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Using mixed precision training (AMP)")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'plateau').lower()
        
        if scheduler_name == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        elif scheduler_name == 'cosine':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-7
            )
        else:
            return None
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Print progress
            if (batch_idx + 1) % self.config.get('log_interval', 50) == 0:
                print(f"    Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # For detailed metrics
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        for images, labels, _ in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probabilities.extend(outputs.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        # Calculate additional metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # True positives, false positives, etc.
        tp = np.sum((all_predictions == 1) & (all_labels == 1))
        fp = np.sum((all_predictions == 1) & (all_labels == 0))
        tn = np.sum((all_predictions == 0) & (all_labels == 0))
        fn = np.sum((all_predictions == 0) & (all_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self, num_epochs: int) -> Dict:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history dictionary
        """
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config.get('batch_size', 32)}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()
            
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            print("-" * 40)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_metrics = self.validate()
            
            # Update learning rate scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  Precision: {val_metrics['precision']:.4f} | "
                  f"Recall: {val_metrics['recall']:.4f} | "
                  f"F1: {val_metrics['f1_score']:.4f}")
            print(f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            
            # Check for improvement
            improved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                improved = True
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_model.pth', val_metrics)
                print(f"  ✓ New best model saved!")
            else:
                self.patience_counter += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', val_metrics)
            
            # Early stopping
            patience = self.config.get('patience', 15)
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print("=" * 60)
        
        # Save final model and history
        self.save_checkpoint('final_model.pth', val_metrics)
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, filename: str, metrics: Dict = None):
        """Save model checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'metrics': metrics,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint.get('history', self.history)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def save_history(self):
        """Save training history to JSON."""
        filepath = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {filepath}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DSH Detector CNN')
    
    # Data arguments
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to the CSV file with image metadata')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing FITS files')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='full',
                        choices=['full', 'lite'],
                        help='Model architecture (full or lite)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for regularization')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='Learning rate scheduler')
    
    # Data arguments
    parser.add_argument('--image_size', type=int, default=64,
                        help='Input image size')
    parser.add_argument('--negative_ratio', type=float, default=1.0,
                        help='Ratio of negative to positive samples')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Training options
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='How often to log batch progress')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        include_negatives=True,
        negative_ratio=args.negative_ratio,
        augment_train=not args.no_augmentation
    )
    
    # Create model
    print("\nCreating model...")
    if args.model == 'full':
        model = DSHDetectorCNN(dropout_rate=args.dropout)
    else:
        model = DSHDetectorCNNLite(dropout_rate=args.dropout)
    
    print(f"Model: {args.model}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Create config dictionary
    config = vars(args)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    history = trainer.train(num_epochs=args.epochs)
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    # Load best model for testing
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predictions = (outputs > 0.5).float()
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)
    
    test_accuracy = test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
