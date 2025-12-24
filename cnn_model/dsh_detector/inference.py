"""
DSH Detector Inference Module
==============================
Inference utilities for the Dust Scattering Halo detection model.

Features:
- Single image prediction
- Batch prediction
- Sliding window detection for survey images
- Confidence-based categorization
- Detection visualization


Usage:
    # Single image inference
    python inference.py --image path/to/image.fits --model checkpoints/best_model.pth
    
    # Survey sliding window detection
    python inference.py --survey path/to/survey.fits --model checkpoints/best_model.pth --mode sliding
"""

import os
import sys
import argparse
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from astropy.io import fits

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dsh_cnn import DSHDetectorCNN, DSHDetectorCNNLite


@dataclass
class Detection:
    """Data class for a single detection result."""
    x: int                    # X coordinate (left edge of window)
    y: int                    # Y coordinate (top edge of window)
    probability: float        # Detection probability [0, 1]
    category: str            # Confidence category
    window_size: int         # Size of the detection window
    
    def __repr__(self):
        return (f"Detection(x={self.x}, y={self.y}, "
                f"prob={self.probability:.3f}, category='{self.category}')")


class DSHInference:
    """
    Inference class for DSH detection.
    
    Provides methods for:
    - Loading trained models
    - Single image prediction
    - Batch prediction
    - Sliding window survey detection
    """
    
    # Confidence thresholds for categorization
    CONFIDENCE_THRESHOLDS = {
        'DEFINITE_HALO': 0.85,
        'PROBABLE_HALO': 0.65,
        'POSSIBLE_HALO': 0.45,
        'UNLIKELY_HALO': 0.25,
        'NO_HALO': 0.0
    }
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'full',
        device: Optional[str] = None,
        image_size: int = 64
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model checkpoint
            model_type: 'full' or 'lite'
            device: Device to use ('cuda', 'cpu', or None for auto)
            image_size: Expected input image size
        """
        self.image_size = image_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path, model_type)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def _load_model(self, model_path: str, model_type: str) -> nn.Module:
        """Load the trained model from checkpoint."""
        # Create model architecture
        if model_type == 'full':
            model = DSHDetectorCNN()
        else:
            model = DSHDetectorCNNLite()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    def preprocess_image(
        self,
        image: np.ndarray,
        log_transform: bool = True,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: Input image as numpy array
            log_transform: Apply log transformation
            normalize: Normalize to [0, 1]
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Handle NaN and Inf
        image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        
        # Resize if necessary
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            from scipy.ndimage import zoom
            zoom_h = self.image_size / image.shape[0]
            zoom_w = self.image_size / image.shape[1]
            image = zoom(image, (zoom_h, zoom_w), order=1)
        
        # Log transform
        if log_transform:
            image = np.log1p(np.clip(image, 0, None))
        
        # Normalize
        if normalize:
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
            else:
                image = np.zeros_like(image)
        
        # Convert to tensor: (1, 1, H, W)
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def get_category(self, probability: float) -> str:
        """Get confidence category from probability."""
        if probability >= self.CONFIDENCE_THRESHOLDS['DEFINITE_HALO']:
            return 'DEFINITE_HALO'
        elif probability >= self.CONFIDENCE_THRESHOLDS['PROBABLE_HALO']:
            return 'PROBABLE_HALO'
        elif probability >= self.CONFIDENCE_THRESHOLDS['POSSIBLE_HALO']:
            return 'POSSIBLE_HALO'
        elif probability >= self.CONFIDENCE_THRESHOLDS['UNLIKELY_HALO']:
            return 'UNLIKELY_HALO'
        else:
            return 'NO_HALO'
    
    @torch.no_grad()
    def predict(self, image: Union[np.ndarray, str]) -> Tuple[float, str]:
        """
        Predict DSH probability for a single image.
        
        Args:
            image: Either numpy array or path to FITS file
            
        Returns:
            Tuple of (probability, category)
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = self.load_fits(image)
        
        # Preprocess
        tensor = self.preprocess_image(image)
        
        # Inference
        output = self.model(tensor)
        probability = output.item()
        category = self.get_category(probability)
        
        return probability, category
    
    @torch.no_grad()
    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[float, str]]:
        """
        Predict DSH probability for a batch of images.
        
        Args:
            images: List of numpy arrays
            
        Returns:
            List of (probability, category) tuples
        """
        # Preprocess all images
        tensors = []
        for img in images:
            tensor = self.preprocess_image(img)
            tensors.append(tensor)
        
        # Stack into batch
        batch = torch.cat(tensors, dim=0)
        
        # Inference
        outputs = self.model(batch)
        
        # Process results
        results = []
        for prob in outputs.cpu().numpy().flatten():
            category = self.get_category(prob)
            results.append((float(prob), category))
        
        return results
    
    def load_fits(self, path: str) -> np.ndarray:
        """Load a FITS file and return the image data."""
        with fits.open(path) as hdul:
            image = hdul[0].data
            if image is None and len(hdul) > 1:
                image = hdul[1].data
            return image.astype(np.float32)
    
    @torch.no_grad()
    def sliding_window_detection(
        self,
        survey_image: Union[np.ndarray, str],
        window_size: int = 64,
        stride: int = 32,
        threshold: float = 0.45,
        batch_size: int = 64,
        verbose: bool = True
    ) -> List[Detection]:
        """
        Detect DSHs in a large survey image using sliding window.
        
        Args:
            survey_image: Survey image (numpy array or path to FITS)
            window_size: Size of the sliding window
            stride: Step size between windows
            threshold: Minimum probability to report as detection
            batch_size: Number of windows to process at once
            verbose: Print progress information
            
        Returns:
            List of Detection objects for all detections above threshold
        """
        # Load image if path
        if isinstance(survey_image, str):
            survey_image = self.load_fits(survey_image)
            if verbose:
                print(f"Loaded survey image: {survey_image.shape}")
        
        # Handle NaN/Inf
        survey_image = np.nan_to_num(survey_image.astype(np.float32), nan=0.0)
        
        height, width = survey_image.shape
        detections = []
        
        # Calculate number of windows
        n_windows_h = (height - window_size) // stride + 1
        n_windows_w = (width - window_size) // stride + 1
        total_windows = n_windows_h * n_windows_w
        
        if verbose:
            print(f"Survey size: {height} x {width}")
            print(f"Window size: {window_size}, Stride: {stride}")
            print(f"Total windows to process: {total_windows}")
        
        # Extract all windows
        windows = []
        positions = []
        
        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                window = survey_image[y:y + window_size, x:x + window_size]
                windows.append(window)
                positions.append((x, y))
        
        # Process in batches
        all_probabilities = []
        
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i + batch_size]
            
            # Preprocess batch
            tensors = []
            for window in batch_windows:
                # Apply same preprocessing as training
                window = np.log1p(np.clip(window, 0, None))
                w_min, w_max = window.min(), window.max()
                if w_max > w_min:
                    window = (window - w_min) / (w_max - w_min)
                else:
                    window = np.zeros_like(window)
                
                tensor = torch.from_numpy(window).unsqueeze(0).unsqueeze(0)
                tensors.append(tensor)
            
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)
            
            # Inference
            outputs = self.model(batch_tensor)
            probs = outputs.cpu().numpy().flatten()
            all_probabilities.extend(probs)
            
            if verbose and (i // batch_size) % 10 == 0:
                print(f"  Processed {min(i + batch_size, len(windows))}/{len(windows)} windows")
        
        # Collect detections above threshold
        for (x, y), prob in zip(positions, all_probabilities):
            if prob >= threshold:
                category = self.get_category(prob)
                detection = Detection(
                    x=x,
                    y=y,
                    probability=float(prob),
                    category=category,
                    window_size=window_size
                )
                detections.append(detection)
        
        if verbose:
            print(f"\nFound {len(detections)} detections above threshold {threshold}")
            
            # Summary by category
            categories = {}
            for d in detections:
                categories[d.category] = categories.get(d.category, 0) + 1
            print("Detections by category:")
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                print(f"  {cat}: {count}")
        
        return detections
    
    def non_max_suppression(
        self,
        detections: List[Detection],
        iou_threshold: float = 0.3
    ) -> List[Detection]:
        """
        Apply non-maximum suppression to remove overlapping detections.
        
        Args:
            detections: List of Detection objects
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Sort by probability (descending)
        detections = sorted(detections, key=lambda d: -d.probability)
        
        kept = []
        
        while detections:
            # Keep the highest probability detection
            best = detections.pop(0)
            kept.append(best)
            
            # Remove overlapping detections
            remaining = []
            for det in detections:
                iou = self._compute_iou(best, det)
                if iou < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return kept
    
    def _compute_iou(self, det1: Detection, det2: Detection) -> float:
        """Compute Intersection over Union between two detections."""
        # Box 1
        x1_1, y1_1 = det1.x, det1.y
        x2_1, y2_1 = det1.x + det1.window_size, det1.y + det1.window_size
        
        # Box 2
        x1_2, y1_2 = det2.x, det2.y
        x2_2, y2_2 = det2.x + det2.window_size, det2.y + det2.window_size
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = det1.window_size ** 2
        area2 = det2.window_size ** 2
        union = area1 + area2 - intersection
        
        return intersection / union


class SurveyScanner:
    """
    High-level survey scanning utility.
    
    Provides easy-to-use interface for scanning eROSITA or similar
    survey data for dust scattering halos.
    """
    
    def __init__(self, model_path: str, model_type: str = 'full'):
        """
        Initialize the survey scanner.
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: 'full' or 'lite'
        """
        self.inference = DSHInference(model_path, model_type)
    
    def scan_survey(
        self,
        survey_path: str,
        output_path: Optional[str] = None,
        window_size: int = 64,
        stride: int = 32,
        threshold: float = 0.45,
        apply_nms: bool = True,
        nms_threshold: float = 0.3
    ) -> List[Detection]:
        """
        Scan a survey image for DSHs.
        
        Args:
            survey_path: Path to the survey FITS file
            output_path: Optional path to save results (JSON)
            window_size: Detection window size
            stride: Sliding window stride
            threshold: Detection probability threshold
            apply_nms: Whether to apply non-maximum suppression
            nms_threshold: IoU threshold for NMS
            
        Returns:
            List of detections
        """
        print(f"\n{'='*60}")
        print(f"Scanning Survey: {survey_path}")
        print(f"{'='*60}")
        
        # Run sliding window detection
        detections = self.inference.sliding_window_detection(
            survey_path,
            window_size=window_size,
            stride=stride,
            threshold=threshold
        )
        
        # Apply NMS if requested
        if apply_nms and detections:
            print(f"\nApplying non-maximum suppression (IoU threshold: {nms_threshold})...")
            original_count = len(detections)
            detections = self.inference.non_max_suppression(detections, nms_threshold)
            print(f"Reduced from {original_count} to {len(detections)} detections")
        
        # Save results if path provided
        if output_path and detections:
            self._save_results(detections, output_path)
        
        return detections
    
    def _save_results(self, detections: List[Detection], output_path: str):
        """Save detection results to JSON."""
        import json
        
        results = {
            'num_detections': len(detections),
            'detections': [
                {
                    'x': d.x,
                    'y': d.y,
                    'probability': d.probability,
                    'category': d.category,
                    'window_size': d.window_size
                }
                for d in detections
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='DSH Detection Inference')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='full',
                        choices=['full', 'lite'],
                        help='Model architecture type')
    
    # Input options
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single FITS image')
    parser.add_argument('--survey', type=str, default=None,
                        help='Path to survey FITS image for sliding window')
    
    # Sliding window options
    parser.add_argument('--window_size', type=int, default=64,
                        help='Sliding window size')
    parser.add_argument('--stride', type=int, default=32,
                        help='Sliding window stride')
    parser.add_argument('--threshold', type=float, default=0.45,
                        help='Detection probability threshold')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results (JSON)')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = DSHInference(args.model, args.model_type)
    
    if args.image:
        # Single image prediction
        print(f"\nPredicting on: {args.image}")
        prob, category = inference.predict(args.image)
        print(f"Probability: {prob:.4f}")
        print(f"Category: {category}")
        
    elif args.survey:
        # Survey scanning
        scanner = SurveyScanner(args.model, args.model_type)
        detections = scanner.scan_survey(
            args.survey,
            output_path=args.output,
            window_size=args.window_size,
            stride=args.stride,
            threshold=args.threshold
        )
        
        # Print top detections
        if detections:
            print("\nTop 10 detections:")
            for i, det in enumerate(sorted(detections, key=lambda d: -d.probability)[:10]):
                print(f"  {i+1}. {det}")
    else:
        print("Please specify either --image or --survey")


if __name__ == '__main__':
    main()
