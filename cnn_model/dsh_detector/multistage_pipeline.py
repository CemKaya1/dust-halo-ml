"""
DSH Multi-Stage Detection Pipeline
===================================
Advanced detection system for eROSITA survey scanning with:
- Stage 1: Initial 64x64 sliding window detection
- Stage 2: Partial halo detection and flagging
- Stage 3: Multi-scale investigation of candidates

This handles the real-world problem of halos being cut off at window boundaries.

"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import torch
import torch.nn as nn

# Will import from local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DetectionType(Enum):
    """Classification of detection types."""
    FULL_HALO = "full_halo"           # Complete halo visible
    PARTIAL_TOP = "partial_top"        # Halo cut off at top
    PARTIAL_BOTTOM = "partial_bottom"  # Halo cut off at bottom
    PARTIAL_LEFT = "partial_left"      # Halo cut off at left
    PARTIAL_RIGHT = "partial_right"    # Halo cut off at right
    PARTIAL_CORNER = "partial_corner"  # Halo in corner (2 edges)
    UNCERTAIN = "uncertain"            # Needs investigation


@dataclass
class CandidateDetection:
    """Extended detection with partial halo information."""
    x: int
    y: int
    probability: float
    category: str
    window_size: int
    detection_type: DetectionType = DetectionType.FULL_HALO
    edge_scores: Dict[str, float] = field(default_factory=dict)  # Intensity at each edge
    needs_investigation: bool = False
    investigation_priority: int = 0  # Higher = more urgent
    related_detections: List[int] = field(default_factory=list)  # IDs of nearby detections
    multi_scale_results: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'x': self.x,
            'y': self.y,
            'probability': self.probability,
            'category': self.category,
            'window_size': self.window_size,
            'detection_type': self.detection_type.value,
            'edge_scores': self.edge_scores,
            'needs_investigation': self.needs_investigation,
            'investigation_priority': self.investigation_priority,
            'related_detections': self.related_detections,
            'multi_scale_results': self.multi_scale_results
        }


class PartialHaloAnalyzer:
    """
    Analyzes detection windows for signs of partial/cut-off halos.
    
    A partial halo will show:
    - Arc-like structure near edges
    - Higher intensity at edges vs center gradient
    - Asymmetric brightness distribution
    """
    
    def __init__(self, edge_margin: int = 8):
        """
        Args:
            edge_margin: Pixels from edge to analyze for partial halos
        """
        self.edge_margin = edge_margin
    
    def analyze_edges(self, image: np.ndarray) -> Dict[str, float]:
        """
        Compute intensity scores at each edge.
        High edge scores suggest halo extends beyond window.
        
        Args:
            image: 2D numpy array (H, W)
            
        Returns:
            Dictionary with edge intensity scores
        """
        h, w = image.shape
        m = self.edge_margin
        
        # Extract edge regions
        top_region = image[:m, :]
        bottom_region = image[-m:, :]
        left_region = image[:, :m]
        right_region = image[:, -m:]
        
        # Center region for comparison
        center_region = image[h//4:3*h//4, w//4:3*w//4]
        
        # Compute mean intensities
        edge_scores = {
            'top': float(np.mean(top_region)),
            'bottom': float(np.mean(bottom_region)),
            'left': float(np.mean(left_region)),
            'right': float(np.mean(right_region)),
            'center': float(np.mean(center_region))
        }
        
        # Compute edge-to-center ratios (high ratio = possible partial halo)
        center_val = edge_scores['center'] + 1e-8  # Avoid division by zero
        edge_scores['top_ratio'] = edge_scores['top'] / center_val
        edge_scores['bottom_ratio'] = edge_scores['bottom'] / center_val
        edge_scores['left_ratio'] = edge_scores['left'] / center_val
        edge_scores['right_ratio'] = edge_scores['right'] / center_val
        
        return edge_scores
    
    def classify_partial_type(
        self, 
        edge_scores: Dict[str, float],
        ratio_threshold: float = 0.7
    ) -> Tuple[DetectionType, bool]:
        """
        Classify the type of partial halo based on edge analysis.
        
        Args:
            edge_scores: Output from analyze_edges()
            ratio_threshold: Minimum edge/center ratio to flag as partial
            
        Returns:
            Tuple of (DetectionType, needs_investigation)
        """
        high_edges = []
        
        for edge in ['top', 'bottom', 'left', 'right']:
            if edge_scores.get(f'{edge}_ratio', 0) > ratio_threshold:
                high_edges.append(edge)
        
        if len(high_edges) == 0:
            return DetectionType.FULL_HALO, False
        elif len(high_edges) >= 2:
            return DetectionType.PARTIAL_CORNER, True
        elif 'top' in high_edges:
            return DetectionType.PARTIAL_TOP, True
        elif 'bottom' in high_edges:
            return DetectionType.PARTIAL_BOTTOM, True
        elif 'left' in high_edges:
            return DetectionType.PARTIAL_LEFT, True
        elif 'right' in high_edges:
            return DetectionType.PARTIAL_RIGHT, True
        else:
            return DetectionType.UNCERTAIN, True


class MultiScaleInvestigator:
    """
    Investigates candidate detections at multiple scales.
    
    For partial halo candidates:
    1. Zoom out (larger window) to see full structure
    2. Check neighboring windows
    3. Combine evidence from multiple scales
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        base_window_size: int = 64
    ):
        self.model = model
        self.device = device
        self.base_window_size = base_window_size
        
        # Scale factors to investigate
        self.scale_factors = [0.5, 1.0, 1.5, 2.0]  # 32, 64, 96, 128 pixels
    
    def investigate_candidate(
        self,
        survey_image: np.ndarray,
        candidate: CandidateDetection,
        context_margin: int = 32
    ) -> Dict[str, any]:
        """
        Perform multi-scale investigation of a candidate.
        
        Args:
            survey_image: Full survey image
            candidate: Candidate detection to investigate
            context_margin: Extra pixels around candidate to consider
            
        Returns:
            Investigation results with multi-scale probabilities
        """
        results = {
            'scales': {},
            'max_probability': 0.0,
            'best_scale': 1.0,
            'recommended_action': 'keep'
        }
        
        h, w = survey_image.shape
        cx = candidate.x + candidate.window_size // 2  # Center x
        cy = candidate.y + candidate.window_size // 2  # Center y
        
        for scale in self.scale_factors:
            window_size = int(self.base_window_size * scale)
            half_size = window_size // 2
            
            # Extract window centered on candidate
            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(w, cx + half_size)
            y2 = min(h, cy + half_size)
            
            window = survey_image[y1:y2, x1:x2]
            
            # Skip if window is too small (near survey edge)
            if window.shape[0] < window_size * 0.5 or window.shape[1] < window_size * 0.5:
                continue
            
            # Resize to model input size and predict
            window_resized = self._resize_window(window, self.base_window_size)
            prob = self._predict(window_resized)
            
            results['scales'][f'scale_{scale}'] = {
                'window_size': window_size,
                'probability': prob,
                'position': (x1, y1, x2, y2)
            }
            
            if prob > results['max_probability']:
                results['max_probability'] = prob
                results['best_scale'] = scale
        
        # Determine recommendation
        if results['max_probability'] >= 0.85:
            results['recommended_action'] = 'confirm_halo'
        elif results['max_probability'] >= 0.65:
            results['recommended_action'] = 'probable_halo'
        elif results['max_probability'] >= 0.45:
            results['recommended_action'] = 'manual_review'
        else:
            results['recommended_action'] = 'likely_false_positive'
        
        return results
    
    def _resize_window(self, window: np.ndarray, target_size: int) -> np.ndarray:
        """Resize window to target size."""
        from scipy.ndimage import zoom
        
        h, w = window.shape
        zoom_h = target_size / h
        zoom_w = target_size / w
        return zoom(window, (zoom_h, zoom_w), order=1)
    
    @torch.no_grad()
    def _predict(self, image: np.ndarray) -> float:
        """Run model prediction on a single image."""
        # Preprocess
        image = np.log1p(np.clip(image, 0, None))
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
        
        # To tensor
        tensor = torch.from_numpy(image.astype(np.float32))
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Predict
        output = self.model(tensor)
        return float(output.item())


class MultiStagePipeline:
    """
    Complete multi-stage detection pipeline for eROSITA surveys.
    
    Stage 1: Fast sliding window scan
    Stage 2: Partial halo analysis and flagging
    Stage 3: Multi-scale investigation of candidates
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'full',
        device: Optional[str] = None
    ):
        from models.dsh_cnn import DSHDetectorCNN, DSHDetectorCNNLite
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        if model_type == 'full':
            self.model = DSHDetectorCNN()
        else:
            self.model = DSHDetectorCNNLite()
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize analyzers
        self.partial_analyzer = PartialHaloAnalyzer()
        self.multi_scale_investigator = MultiScaleInvestigator(
            self.model, self.device
        )
        
        print(f"Multi-stage pipeline initialized on {self.device}")
    
    def scan_survey(
        self,
        survey_image: np.ndarray,
        window_size: int = 64,
        stride: int = 32,
        initial_threshold: float = 0.35,  # Lower threshold to catch partials
        final_threshold: float = 0.45,
        batch_size: int = 64,
        investigate_partials: bool = True,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Run complete multi-stage detection on a survey image.
        
        Args:
            survey_image: Full survey image (2D numpy array)
            window_size: Sliding window size
            stride: Stride between windows
            initial_threshold: Threshold for Stage 1 (lower to catch partials)
            final_threshold: Final threshold after investigation
            batch_size: Batch size for inference
            investigate_partials: Whether to run Stage 3 on partial detections
            verbose: Print progress
            
        Returns:
            Complete results dictionary
        """
        results = {
            'survey_shape': survey_image.shape,
            'parameters': {
                'window_size': window_size,
                'stride': stride,
                'initial_threshold': initial_threshold,
                'final_threshold': final_threshold
            },
            'stage1_candidates': [],
            'stage2_partial_flags': [],
            'stage3_investigated': [],
            'final_detections': [],
            'statistics': {}
        }
        
        if verbose:
            print("=" * 60)
            print("STAGE 1: Initial Sliding Window Scan")
            print("=" * 60)
        
        # Stage 1: Initial scan
        stage1_candidates = self._stage1_scan(
            survey_image, window_size, stride, 
            initial_threshold, batch_size, verbose
        )
        results['stage1_candidates'] = [c.to_dict() for c in stage1_candidates]
        
        if verbose:
            print(f"\nStage 1 complete: {len(stage1_candidates)} candidates found")
            print("\n" + "=" * 60)
            print("STAGE 2: Partial Halo Analysis")
            print("=" * 60)
        
        # Stage 2: Analyze for partial halos
        stage2_candidates = self._stage2_analyze(
            survey_image, stage1_candidates, verbose
        )
        
        partial_count = sum(1 for c in stage2_candidates if c.needs_investigation)
        results['stage2_partial_flags'] = [
            c.to_dict() for c in stage2_candidates if c.needs_investigation
        ]
        
        if verbose:
            print(f"\nStage 2 complete: {partial_count} candidates flagged for investigation")
        
        # Stage 3: Multi-scale investigation
        if investigate_partials:
            if verbose:
                print("\n" + "=" * 60)
                print("STAGE 3: Multi-Scale Investigation")
                print("=" * 60)
            
            stage3_candidates = self._stage3_investigate(
                survey_image, stage2_candidates, verbose
            )
            results['stage3_investigated'] = [
                c.to_dict() for c in stage3_candidates 
                if c.multi_scale_results
            ]
        else:
            stage3_candidates = stage2_candidates
        
        # Final filtering
        final_detections = [
            c for c in stage3_candidates 
            if c.probability >= final_threshold or 
               c.multi_scale_results.get('max_probability', 0) >= final_threshold
        ]
        
        # Apply non-maximum suppression
        final_detections = self._non_max_suppression(final_detections)
        results['final_detections'] = [c.to_dict() for c in final_detections]
        
        # Statistics
        results['statistics'] = {
            'total_windows_scanned': ((survey_image.shape[0] - window_size) // stride + 1) * 
                                     ((survey_image.shape[1] - window_size) // stride + 1),
            'stage1_candidates': len(stage1_candidates),
            'stage2_partial_flags': partial_count,
            'final_detections': len(final_detections),
            'detection_types': self._count_detection_types(final_detections)
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("FINAL RESULTS")
            print("=" * 60)
            print(f"Total detections: {len(final_detections)}")
            print(f"Detection types: {results['statistics']['detection_types']}")
        
        return results
    
    @torch.no_grad()
    def _stage1_scan(
        self,
        survey_image: np.ndarray,
        window_size: int,
        stride: int,
        threshold: float,
        batch_size: int,
        verbose: bool
    ) -> List[CandidateDetection]:
        """Stage 1: Fast sliding window scan."""
        
        survey_image = np.nan_to_num(survey_image.astype(np.float32), nan=0.0)
        height, width = survey_image.shape
        candidates = []
        
        # Extract all windows
        windows = []
        positions = []
        
        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                window = survey_image[y:y + window_size, x:x + window_size]
                windows.append(window)
                positions.append((x, y))
        
        if verbose:
            print(f"Total windows to scan: {len(windows)}")
        
        # Process in batches
        all_probabilities = []
        
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i + batch_size]
            
            tensors = []
            for window in batch_windows:
                # Preprocess
                w = np.log1p(np.clip(window, 0, None))
                w_min, w_max = w.min(), w.max()
                if w_max > w_min:
                    w = (w - w_min) / (w_max - w_min)
                else:
                    w = np.zeros_like(w)
                
                tensor = torch.from_numpy(w.astype(np.float32))
                tensor = tensor.unsqueeze(0).unsqueeze(0)
                tensors.append(tensor)
            
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)
            outputs = self.model(batch_tensor)
            probs = outputs.cpu().numpy().flatten()
            all_probabilities.extend(probs)
            
            if verbose and (i // batch_size) % 20 == 0:
                print(f"  Scanned {min(i + batch_size, len(windows))}/{len(windows)} windows")
        
        # Create candidates above threshold
        for (x, y), prob in zip(positions, all_probabilities):
            if prob >= threshold:
                category = self._get_category(prob)
                candidate = CandidateDetection(
                    x=x, y=y,
                    probability=float(prob),
                    category=category,
                    window_size=window_size
                )
                candidates.append(candidate)
        
        return candidates
    
    def _stage2_analyze(
        self,
        survey_image: np.ndarray,
        candidates: List[CandidateDetection],
        verbose: bool
    ) -> List[CandidateDetection]:
        """Stage 2: Analyze candidates for partial halos."""
        
        for i, candidate in enumerate(candidates):
            # Extract window
            x, y = candidate.x, candidate.y
            ws = candidate.window_size
            window = survey_image[y:y+ws, x:x+ws]
            
            # Analyze edges
            edge_scores = self.partial_analyzer.analyze_edges(window)
            candidate.edge_scores = edge_scores
            
            # Classify partial type
            detection_type, needs_investigation = self.partial_analyzer.classify_partial_type(
                edge_scores
            )
            candidate.detection_type = detection_type
            candidate.needs_investigation = needs_investigation
            
            # Set investigation priority
            if needs_investigation:
                # Higher probability + partial = higher priority
                candidate.investigation_priority = int(candidate.probability * 100)
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  Analyzed {i + 1}/{len(candidates)} candidates")
        
        return candidates
    
    def _stage3_investigate(
        self,
        survey_image: np.ndarray,
        candidates: List[CandidateDetection],
        verbose: bool
    ) -> List[CandidateDetection]:
        """Stage 3: Multi-scale investigation of flagged candidates."""
        
        # Sort by priority
        to_investigate = [c for c in candidates if c.needs_investigation]
        to_investigate.sort(key=lambda c: -c.investigation_priority)
        
        if verbose:
            print(f"Investigating {len(to_investigate)} candidates")
        
        for i, candidate in enumerate(to_investigate):
            results = self.multi_scale_investigator.investigate_candidate(
                survey_image, candidate
            )
            candidate.multi_scale_results = results
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Investigated {i + 1}/{len(to_investigate)}")
        
        return candidates
    
    def _non_max_suppression(
        self,
        detections: List[CandidateDetection],
        iou_threshold: float = 0.3
    ) -> List[CandidateDetection]:
        """Remove overlapping detections."""
        if not detections:
            return []
        
        # Sort by probability
        detections = sorted(detections, key=lambda d: -d.probability)
        
        kept = []
        while detections:
            best = detections.pop(0)
            kept.append(best)
            
            remaining = []
            for det in detections:
                iou = self._compute_iou(best, det)
                if iou < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return kept
    
    def _compute_iou(self, d1: CandidateDetection, d2: CandidateDetection) -> float:
        """Compute IoU between two detections."""
        x1_1, y1_1 = d1.x, d1.y
        x2_1, y2_1 = d1.x + d1.window_size, d1.y + d1.window_size
        
        x1_2, y1_2 = d2.x, d2.y
        x2_2, y2_2 = d2.x + d2.window_size, d2.y + d2.window_size
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = d1.window_size ** 2
        area2 = d2.window_size ** 2
        union = area1 + area2 - intersection
        
        return intersection / union
    
    def _get_category(self, probability: float) -> str:
        """Get confidence category."""
        if probability >= 0.85:
            return 'DEFINITE_HALO'
        elif probability >= 0.65:
            return 'PROBABLE_HALO'
        elif probability >= 0.45:
            return 'POSSIBLE_HALO'
        elif probability >= 0.25:
            return 'UNLIKELY_HALO'
        else:
            return 'NO_HALO'
    
    def _count_detection_types(self, detections: List[CandidateDetection]) -> Dict[str, int]:
        """Count detections by type."""
        counts = {}
        for d in detections:
            dtype = d.detection_type.value
            counts[dtype] = counts.get(dtype, 0) + 1
        return counts
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")


# ============================================================
# Convenience functions for quick usage
# ============================================================

def scan_erosita_survey(
    survey_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    window_size: int = 64,
    stride: int = 32
) -> Dict:
    """
    Quick function to scan an eROSITA survey.
    
    Args:
        survey_path: Path to survey FITS file
        model_path: Path to trained model
        output_path: Optional path to save results
        window_size: Detection window size
        stride: Sliding window stride
        
    Returns:
        Detection results dictionary
    """
    from astropy.io import fits
    
    # Load survey
    print(f"Loading survey: {survey_path}")
    with fits.open(survey_path) as hdul:
        survey_image = hdul[0].data
        if survey_image is None and len(hdul) > 1:
            survey_image = hdul[1].data
    
    print(f"Survey shape: {survey_image.shape}")
    
    # Initialize pipeline
    pipeline = MultiStagePipeline(model_path)
    
    # Run detection
    results = pipeline.scan_survey(
        survey_image,
        window_size=window_size,
        stride=stride
    )
    
    # Save if requested
    if output_path:
        pipeline.save_results(results, output_path)
    
    return results


if __name__ == "__main__":
    print("Multi-Stage DSH Detection Pipeline")
    print("=" * 60)
    print("""
Usage:
    from multistage_pipeline import MultiStagePipeline, scan_erosita_survey
    
    # Quick scan
    results = scan_erosita_survey(
        survey_path='erosita_survey.fits',
        model_path='checkpoints/best_model.pth',
        output_path='detections.json'
    )
    
    # Or with full control
    pipeline = MultiStagePipeline('checkpoints/best_model.pth')
    results = pipeline.scan_survey(
        survey_image,
        window_size=64,
        stride=32,
        initial_threshold=0.35,
        investigate_partials=True
    )
    
    # Results include:
    # - stage1_candidates: All initial detections
    # - stage2_partial_flags: Detections flagged as partial halos
    # - stage3_investigated: Multi-scale investigation results
    # - final_detections: Final filtered detections
    """)
