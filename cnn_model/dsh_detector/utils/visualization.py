"""
DSH Detector Visualization Utilities
=====================================
Tools for visualizing training progress and detection results.

"""

import os
import json
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


def plot_training_history(
    history_path: str,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training history curves.
    
    Args:
        history_path: Path to training_history.json
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DSH Detector Training History', fontsize=14, fontweight='bold')
    
    # Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Learning rate
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Loss vs Accuracy scatter
    ax4 = axes[1, 1]
    scatter = ax4.scatter(
        history['val_loss'],
        history['val_acc'],
        c=epochs,
        cmap='viridis',
        s=50
    )
    ax4.set_xlabel('Validation Loss')
    ax4.set_ylabel('Validation Accuracy')
    ax4.set_title('Loss vs Accuracy (colored by epoch)')
    plt.colorbar(scatter, ax=ax4, label='Epoch')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_sample_predictions(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    n_samples: int = 16,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot a grid of sample predictions.
    
    Args:
        images: Array of images (N, H, W) or (N, 1, H, W)
        labels: True labels (N,)
        predictions: Predicted labels (N,)
        probabilities: Prediction probabilities (N,)
        n_samples: Number of samples to show
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    n_samples = min(n_samples, len(images))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_samples):
        ax = axes[i]
        
        # Get image (handle different shapes)
        img = images[i]
        if img.ndim == 3:
            img = img.squeeze(0)
        
        # Display image
        ax.imshow(img, cmap='viridis', origin='lower')
        
        # Color based on correct/incorrect
        true_label = labels[i]
        pred_label = predictions[i]
        prob = probabilities[i]
        
        is_correct = true_label == pred_label
        color = 'green' if is_correct else 'red'
        
        # Title with prediction info
        true_str = 'Halo' if true_label == 1 else 'No Halo'
        pred_str = 'Halo' if pred_label == 1 else 'No Halo'
        
        ax.set_title(f'True: {true_str}\nPred: {pred_str} ({prob:.2f})', 
                    color=color, fontsize=9)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_survey_detections(
    survey_image: np.ndarray,
    detections: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (14, 14)
):
    """
    Visualize detections on a survey image.
    
    Args:
        survey_image: The survey image array
        detections: List of detection dictionaries with x, y, probability, category
        save_path: Optional path to save the figure
        show: Whether to display the plot
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Display survey image with log scaling for better visualization
    img_display = np.log1p(np.clip(survey_image, 0, None))
    ax.imshow(img_display, cmap='viridis', origin='lower')
    
    # Color map for categories
    category_colors = {
        'DEFINITE_HALO': 'red',
        'PROBABLE_HALO': 'orange',
        'POSSIBLE_HALO': 'yellow',
        'UNLIKELY_HALO': 'cyan',
        'NO_HALO': 'white'
    }
    
    # Draw detection boxes
    for det in detections:
        x = det['x'] if isinstance(det, dict) else det.x
        y = det['y'] if isinstance(det, dict) else det.y
        prob = det['probability'] if isinstance(det, dict) else det.probability
        category = det['category'] if isinstance(det, dict) else det.category
        window_size = det.get('window_size', 64) if isinstance(det, dict) else det.window_size
        
        color = category_colors.get(category, 'white')
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x, y), window_size, window_size,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add probability label
        ax.text(
            x + 2, y + window_size - 5,
            f'{prob:.2f}',
            color=color,
            fontsize=8,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5)
        )
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor=color, label=cat, linewidth=2)
        for cat, color in category_colors.items()
        if any((d['category'] if isinstance(d, dict) else d.category) == cat for d in detections)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_title(f'Survey Detections ({len(detections)} total)', fontsize=14)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_confidence_distribution(
    probabilities: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot the distribution of prediction confidence for positive and negative samples.
    
    Args:
        probabilities: Model output probabilities
        labels: True labels
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Separate by true label
    pos_probs = probabilities[labels == 1]
    neg_probs = probabilities[labels == 0]
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(pos_probs, bins=50, alpha=0.7, label='True Halo', color='blue', density=True)
    ax1.hist(neg_probs, bins=50, alpha=0.7, label='True No-Halo', color='red', density=True)
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Density')
    ax1.set_title('Confidence Distribution by True Label')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add threshold lines
    thresholds = [0.25, 0.45, 0.65, 0.85]
    for thresh in thresholds:
        ax1.axvline(thresh, color='gray', linestyle='--', alpha=0.5)
    
    # Box plot
    ax2 = axes[1]
    data = [pos_probs, neg_probs]
    bp = ax2.boxplot(data, labels=['True Halo', 'True No-Halo'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax2.set_ylabel('Predicted Probability')
    ax2.set_title('Confidence Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Model Confidence Analysis', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        labels: True labels
        predictions: Predicted labels
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    # Compute confusion matrix
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot
    im = ax.imshow(cm, cmap='Blues')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            total = cm.sum()
            percentage = value / total * 100
            text = f'{value}\n({percentage:.1f}%)'
            ax.text(j, i, text, ha='center', va='center', fontsize=14)
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Halo', 'Halo'])
    ax.set_yticklabels(['No Halo', 'Halo'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Calculate and display metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1 Score: {f1:.3f}'
    ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("DSH Detector Visualization Utilities")
    print("=====================================")
    print("\nAvailable functions:")
    print("  - plot_training_history(history_path)")
    print("  - plot_sample_predictions(images, labels, predictions, probs)")
    print("  - plot_survey_detections(survey_image, detections)")
    print("  - plot_confidence_distribution(probabilities, labels)")
    print("  - plot_confusion_matrix(labels, predictions)")
