"""
DSH Multi-Stage Detection Pipeline - Complete
==============================================
Unified pipeline combining sliding window detection, arc analysis,
clustering, and large halo inference.

Architecture:
    Stage 1: Multi-scale sliding window scan (memory-safe)
    Stage 2: Arc direction analysis (where is the signal in each window?)
    Stage 3: Spatial clustering (group nearby detections)
    Stage 4: Multi-scale investigation (zoom out for partial halos)
    Stage 5: Large halo inference (combine arc patterns)
    Stage 6: Export to investigation database

Key Features:
- Generator-based scanning (constant memory)
- GPU-accelerated operations
- Reflection padding for edge candidates
- Arc pattern analysis for giant halos
- Optimized clustering algorithm

Author: DSH Detection Project - Part 5 (Model Selection & Training)
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Generator
from enum import Enum
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ENUMS AND DATA CLASSES
# ============================================================

class ArcDirection(Enum):
    """Direction where arc/halo structure is detected in window."""
    CENTER = "center"           # Full halo, centered
    TOP = "top"                 # Arc at top edge
    BOTTOM = "bottom"           # Arc at bottom edge
    LEFT = "left"               # Arc at left edge
    RIGHT = "right"             # Arc at right edge
    TOP_LEFT = "top_left"       # Arc in top-left corner
    TOP_RIGHT = "top_right"     # Arc in top-right corner
    BOTTOM_LEFT = "bottom_left" # Arc in bottom-left corner
    BOTTOM_RIGHT = "bottom_right" # Arc in bottom-right corner
    DIFFUSE = "diffuse"         # No clear direction


class CandidateType(Enum):
    """Type of halo candidate."""
    FULL_HALO = "full_halo"
    LARGE_HALO_INFERRED = "large_halo_inferred"
    EDGE_CANDIDATE = "edge_candidate"
    SCATTERED_ARCS = "scattered_arcs"
    UNCERTAIN = "uncertain"


@dataclass
class Detection:
    """Single window detection with arc analysis."""
    x: int
    y: int
    window_size: int
    probability: float
    confidence: str
    arc_direction: ArcDirection = ArcDirection.CENTER
    edge_scores: Dict[str, float] = field(default_factory=dict)
    needs_investigation: bool = False
    cluster_id: Optional[int] = None
    multi_scale_results: Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'x': self.x,
            'y': self.y,
            'window_size': self.window_size,
            'probability': self.probability,
            'confidence': self.confidence,
            'arc_direction': self.arc_direction.value,
            'edge_scores': self.edge_scores,
            'needs_investigation': self.needs_investigation,
            'cluster_id': self.cluster_id,
            'multi_scale_max_prob': self.multi_scale_results.get('max_probability', None)
        }


@dataclass
class HaloCandidate:
    """Inferred halo candidate from clustering."""
    candidate_id: int
    candidate_type: CandidateType
    center_x: float
    center_y: float
    estimated_radius: float
    confidence_score: float
    contributing_detections: List[int]
    arc_pattern: Dict[str, int]
    notes: str
    
    def to_dict(self) -> Dict:
        return {
            'candidate_id': self.candidate_id,
            'candidate_type': self.candidate_type.value,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'estimated_radius': self.estimated_radius,
            'confidence_score': self.confidence_score,
            'num_detections': len(self.contributing_detections),
            'arc_pattern': self.arc_pattern,
            'notes': self.notes
        }


# ============================================================
# ANALYSIS COMPONENTS
# ============================================================

class ArcAnalyzer:
    """
    Analyzes detection windows to determine arc direction.
    
    This is critical for detecting large halos that span multiple windows.
    A window at the edge of a large halo will show an "arc" pattern.
    """
    
    def __init__(self, margin_ratio: float = 0.2):
        """
        Args:
            margin_ratio: Fraction of window to use as edge margin
        """
        self.margin_ratio = margin_ratio
    
    def analyze(self, window: np.ndarray) -> Tuple[ArcDirection, Dict[str, float], bool]:
        """
        Analyze a window for arc direction.
        
        Args:
            window: 2D numpy array
            
        Returns:
            Tuple of (arc_direction, edge_scores, needs_investigation)
        """
        h, w = window.shape
        m_h = int(h * self.margin_ratio)
        m_w = int(w * self.margin_ratio)
        
        # Extract 9 regions (3x3 grid)
        regions = {
            'top_left': window[:m_h, :m_w],
            'top': window[:m_h, m_w:w-m_w],
            'top_right': window[:m_h, w-m_w:],
            'left': window[m_h:h-m_h, :m_w],
            'center': window[m_h:h-m_h, m_w:w-m_w],
            'right': window[m_h:h-m_h, w-m_w:],
            'bottom_left': window[h-m_h:, :m_w],
            'bottom': window[h-m_h:, m_w:w-m_w],
            'bottom_right': window[h-m_h:, w-m_w:]
        }
        
        # Calculate mean intensity for each region
        intensities = {}
        for name, region in regions.items():
            intensities[name] = float(np.mean(region)) if region.size > 0 else 0.0
        
        # Calculate edge-to-center ratios
        center_val = intensities['center'] + 1e-10
        edge_scores = {
            'top': intensities['top'],
            'bottom': intensities['bottom'],
            'left': intensities['left'],
            'right': intensities['right'],
            'center': intensities['center'],
            'top_ratio': intensities['top'] / center_val,
            'bottom_ratio': intensities['bottom'] / center_val,
            'left_ratio': intensities['left'] / center_val,
            'right_ratio': intensities['right'] / center_val
        }
        
        # Determine arc direction
        arc_direction = self._determine_direction(intensities)
        
        # Determine if needs investigation
        needs_investigation = arc_direction not in [ArcDirection.CENTER, ArcDirection.DIFFUSE]
        
        return arc_direction, edge_scores, needs_investigation
    
    def _determine_direction(self, intensities: Dict[str, float]) -> ArcDirection:
        """Determine arc direction from intensity profile."""
        center = intensities['center']
        
        # Calculate scores (how much brighter than center)
        scores = {}
        for region, intensity in intensities.items():
            if region != 'center':
                scores[region] = intensity / (center + 1e-10)
        
        # Find maximum
        max_region = max(scores, key=scores.get)
        max_score = scores[max_region]
        
        # If center is brightest or all similar, it's centered or diffuse
        if max_score < 1.2:
            if center > np.mean(list(intensities.values())):
                return ArcDirection.CENTER
            else:
                return ArcDirection.DIFFUSE
        
        # Map region to direction
        direction_map = {
            'top': ArcDirection.TOP,
            'bottom': ArcDirection.BOTTOM,
            'left': ArcDirection.LEFT,
            'right': ArcDirection.RIGHT,
            'top_left': ArcDirection.TOP_LEFT,
            'top_right': ArcDirection.TOP_RIGHT,
            'bottom_left': ArcDirection.BOTTOM_LEFT,
            'bottom_right': ArcDirection.BOTTOM_RIGHT
        }
        
        return direction_map.get(max_region, ArcDirection.DIFFUSE)


class SpatialClusterer:
    """
    Clusters nearby detections using optimized union-find algorithm.
    
    Detections that are close together likely belong to the same
    large halo structure.
    """
    
    def __init__(self, proximity_threshold: float = 100):
        """
        Args:
            proximity_threshold: Maximum distance to consider detections as related
        """
        self.proximity_threshold = proximity_threshold
    
    def cluster(self, detections: List[Detection]) -> List[List[int]]:
        """
        Cluster detections based on spatial proximity.
        
        Uses union-find with spatial bucketing for efficiency.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of clusters (each cluster is list of detection indices)
        """
        if not detections:
            return []
        
        n = len(detections)
        
        # Extract positions
        positions = np.array([
            (d.x + d.window_size / 2, d.y + d.window_size / 2) 
            for d in detections
        ])
        
        # Union-Find with path compression
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
        
        # Spatial bucketing for O(n) instead of O(n²)
        bucket_size = self.proximity_threshold
        buckets = {}
        
        for i, (x, y) in enumerate(positions):
            bx, by = int(x // bucket_size), int(y // bucket_size)
            
            # Check neighboring buckets
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (bx + dx, by + dy)
                    if key in buckets:
                        for j in buckets[key]:
                            dist = np.sqrt(
                                (positions[i][0] - positions[j][0])**2 + 
                                (positions[i][1] - positions[j][1])**2
                            )
                            if dist < self.proximity_threshold:
                                union(i, j)
            
            # Add to bucket
            key = (bx, by)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(i)
        
        # Collect clusters
        cluster_map = {}
        for i in range(n):
            root = find(i)
            if root not in cluster_map:
                cluster_map[root] = []
            cluster_map[root].append(i)
        
        return list(cluster_map.values())
    
    def analyze_cluster(
        self, 
        detections: List[Detection], 
        cluster_indices: List[int]
    ) -> HaloCandidate:
        """
        Analyze a cluster to infer if it's a large halo.
        
        Looks at the arc patterns to determine halo characteristics.
        """
        cluster_dets = [detections[i] for i in cluster_indices]
        
        # Count arc directions
        arc_pattern = {}
        for det in cluster_dets:
            direction = det.arc_direction.value
            arc_pattern[direction] = arc_pattern.get(direction, 0) + 1
        
        # Calculate cluster center and radius
        positions = [
            (d.x + d.window_size/2, d.y + d.window_size/2) 
            for d in cluster_dets
        ]
        center_x = np.mean([p[0] for p in positions])
        center_y = np.mean([p[1] for p in positions])
        
        distances = [
            np.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2) 
            for p in positions
        ]
        estimated_radius = np.max(distances) + cluster_dets[0].window_size / 2
        
        # Classify cluster
        candidate_type, confidence, notes = self._classify_cluster(
            arc_pattern, len(cluster_dets)
        )
        
        return HaloCandidate(
            candidate_id=-1,  # Assigned later
            candidate_type=candidate_type,
            center_x=center_x,
            center_y=center_y,
            estimated_radius=estimated_radius,
            confidence_score=confidence,
            contributing_detections=cluster_indices,
            arc_pattern=arc_pattern,
            notes=notes
        )
    
    def _classify_cluster(
        self, 
        arc_pattern: Dict[str, int], 
        num_detections: int
    ) -> Tuple[CandidateType, float, str]:
        """
        Classify cluster with STRICT isolation penalty and geometry checks.
        
        KEY RULES:
        1. Single edge detection = NOISE (isolated arc is impossible for real halo)
        2. Single center detection = Valid small halo
        3. Two detections = Weak evidence (unless one is center)
        4. 3+ detections = Check for geometric spread (arcs on opposite sides)
        
        A real large halo MUST have detections on multiple sides!
        """
        center_count = arc_pattern.get('center', 0)
        diffuse_count = arc_pattern.get('diffuse', 0)
        
        # Count edges and corners
        top = arc_pattern.get('top', 0)
        bottom = arc_pattern.get('bottom', 0)
        left = arc_pattern.get('left', 0)
        right = arc_pattern.get('right', 0)
        top_left = arc_pattern.get('top_left', 0)
        top_right = arc_pattern.get('top_right', 0)
        bottom_left = arc_pattern.get('bottom_left', 0)
        bottom_right = arc_pattern.get('bottom_right', 0)
        
        edge_count = top + bottom + left + right
        corner_count = top_left + top_right + bottom_left + bottom_right
        
        # ============================================================
        # RULE 1: Single detection
        # ============================================================
        if num_detections == 1:
            if center_count == 1:
                # Single center = small complete halo (valid)
                return CandidateType.FULL_HALO, 0.9, "Single centered detection - complete small halo"
            else:
                # ISOLATION PENALTY: Single edge/corner = NOISE
                # An isolated arc with no neighbors is mathematically impossible
                # for a real large halo
                return CandidateType.UNCERTAIN, 0.10, \
                    "ISOLATED: Single edge detection with no neighbors - likely noise"
        
        # ============================================================
        # RULE 2: Two detections (weak evidence)
        # ============================================================
        if num_detections == 2:
            if center_count >= 1:
                # One center + one edge = possibly valid
                return CandidateType.FULL_HALO, 0.7, "Center + edge - likely complete halo"
            elif center_count == 2:
                return CandidateType.FULL_HALO, 0.85, "Two centered detections"
            else:
                # Two edges only = weak evidence
                return CandidateType.SCATTERED_ARCS, 0.35, \
                    "Only 2 edge detections - insufficient evidence"
        
        # ============================================================
        # RULE 3: Three or more detections - check geometry
        # ============================================================
        
        # If majority are center, it's a full halo
        if center_count > num_detections * 0.5:
            return CandidateType.FULL_HALO, 0.9, "Majority centered - complete halo"
        
        # If mostly diffuse, it's noise
        if diffuse_count > num_detections * 0.5:
            return CandidateType.UNCERTAIN, 0.2, "Mostly diffuse - likely noise"
        
        # Check for GEOMETRIC SPREAD
        # A real large halo should have arcs on OPPOSITE sides
        has_vertical_spread = (top + top_left + top_right > 0) and (bottom + bottom_left + bottom_right > 0)
        has_horizontal_spread = (left + top_left + bottom_left > 0) and (right + top_right + bottom_right > 0)
        
        # Count how many "sides" have detections
        sides_with_arcs = 0
        if top + top_left + top_right > 0: sides_with_arcs += 1
        if bottom + bottom_left + bottom_right > 0: sides_with_arcs += 1
        if left + top_left + bottom_left > 0: sides_with_arcs += 1
        if right + top_right + bottom_right > 0: sides_with_arcs += 1
        
        # Strong evidence: arcs on 3+ sides, or clear opposite spread
        if sides_with_arcs >= 3 or (has_vertical_spread and has_horizontal_spread):
            return CandidateType.LARGE_HALO_INFERRED, 0.85, \
                f"Strong geometric spread: arcs on {sides_with_arcs} sides"
        
        # Medium evidence: arcs on 2 opposite sides
        if has_vertical_spread or has_horizontal_spread:
            return CandidateType.LARGE_HALO_INFERRED, 0.70, \
                "Moderate spread: arcs on opposite sides"
        
        # Weak evidence: arcs on 2 sides but not opposite
        if sides_with_arcs == 2:
            return CandidateType.SCATTERED_ARCS, 0.45, \
                "Arcs on 2 adjacent sides only - weak evidence"
        
        # All arcs on same side (suspicious)
        if sides_with_arcs == 1:
            return CandidateType.SCATTERED_ARCS, 0.30, \
                "All arcs on same side - likely noise or PSF artifact"
        
        # Fallback
        return CandidateType.UNCERTAIN, 0.35, "Mixed pattern - needs manual review"
    
    def _is_consistent_pattern(self, arc_pattern: Dict[str, int]) -> bool:
        """Check if arc pattern is consistent with a large circular halo."""
        # A large halo would show multiple edge/corner detections
        edge_corner_count = sum(
            v for k, v in arc_pattern.items() 
            if k not in ['center', 'diffuse']
        )
        return edge_corner_count >= 3


class MultiScaleInvestigator:
    """
    Investigates candidates at multiple scales.
    
    Uses GPU-accelerated resizing and reflection padding for edge candidates.
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
        self.scale_factors = [0.5, 1.0, 1.5, 2.0]
    
    @torch.no_grad()
    def investigate(
        self,
        image: np.ndarray,
        detection: Detection
    ) -> Dict[str, any]:
        """
        Multi-scale investigation of a detection.
        
        Zooms out to see if partial halo becomes full halo at larger scale.
        """
        results = {
            'scales': {},
            'max_probability': detection.probability,
            'best_scale': 1.0,
            'used_reflection_padding': False
        }
        
        h, w = image.shape
        cx = detection.x + detection.window_size // 2
        cy = detection.y + detection.window_size // 2
        
        for scale in self.scale_factors:
            window_size = int(self.base_window_size * scale)
            half_size = window_size // 2
            
            # Calculate bounds
            x1, y1 = cx - half_size, cy - half_size
            x2, y2 = cx + half_size, cy + half_size
            
            # Check if needs reflection padding
            needs_padding = x1 < 0 or y1 < 0 or x2 > w or y2 > h
            
            if needs_padding:
                window = self._extract_with_padding(image, cx, cy, half_size)
                results['used_reflection_padding'] = True
            else:
                window = image[y1:y2, x1:x2]
            
            if window is None or window.size == 0:
                continue
            
            # GPU-accelerated prediction
            prob = self._predict(window)
            
            results['scales'][f'scale_{scale}'] = {
                'window_size': window_size,
                'probability': prob,
                'used_padding': needs_padding
            }
            
            if prob > results['max_probability']:
                results['max_probability'] = prob
                results['best_scale'] = scale
        
        return results
    
    def _extract_with_padding(
        self,
        image: np.ndarray,
        cx: int,
        cy: int,
        half_size: int
    ) -> Optional[np.ndarray]:
        """Extract window with reflection padding for edge cases."""
        h, w = image.shape
        
        # Calculate padding needed
        pad_left = max(0, -(cx - half_size))
        pad_right = max(0, (cx + half_size) - w)
        pad_top = max(0, -(cy - half_size))
        pad_bottom = max(0, (cy + half_size) - h)
        
        # Clamp coordinates
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)
        
        valid_region = image[y1:y2, x1:x2]
        
        if valid_region.size == 0:
            return None
        
        # Apply reflection padding
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            return np.pad(
                valid_region,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='reflect'
            )
        
        return valid_region
    
    def _predict(self, window: np.ndarray) -> float:
        """GPU-accelerated prediction with resizing."""
        # Convert to tensor
        tensor = torch.from_numpy(window.astype(np.float32))
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # GPU resize
        if tensor.shape[-1] != self.base_window_size:
            tensor = F.interpolate(
                tensor,
                size=(self.base_window_size, self.base_window_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Preprocess
        tensor = torch.log1p(torch.clamp(tensor, min=0))
        t_min, t_max = tensor.min(), tensor.max()
        if t_max > t_min:
            tensor = (tensor - t_min) / (t_max - t_min)
        
        return float(self.model(tensor).item())


# ============================================================
# MAIN PIPELINE
# ============================================================

class DSHPipeline:
    """
    Complete DSH Detection Pipeline.
    
    Combines all stages:
    1. Multi-scale sliding window scan
    2. Arc direction analysis
    3. Spatial clustering
    4. Multi-scale investigation
    5. Large halo inference
    6. Export results
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'resnet',
        device: Optional[str] = None,
        window_size: int = 64,
        stride: int = 32,
        batch_size: int = 128,
        detection_threshold: float = 0.35,
        cluster_proximity: float = 100,
        min_nonzero_pixels: int = 5
    ):
        """
        Initialize pipeline.
        
        Args:
            model_path: Path to model checkpoint
            model_type: 'resnet', 'resnet_lite', or 'cnn'
            device: 'cuda', 'cpu', or None for auto
            window_size: Base window size
            stride: Stride for sliding window
            batch_size: Batch size for inference
            detection_threshold: Initial detection threshold
            cluster_proximity: Distance threshold for clustering
            min_nonzero_pixels: Minimum non-zero pixels to process window
        """
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path, model_type)
        self.model.eval()
        
        # Parameters
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.detection_threshold = detection_threshold
        self.min_nonzero_pixels = min_nonzero_pixels
        
        # Components
        self.arc_analyzer = ArcAnalyzer()
        self.clusterer = SpatialClusterer(proximity_threshold=cluster_proximity)
        self.investigator = MultiScaleInvestigator(self.model, self.device, window_size)
        
        print(f"DSHPipeline initialized on {self.device}")
        print(f"  Window: {window_size}x{window_size}, Stride: {stride}")
        print(f"  Threshold: {detection_threshold}, Batch: {batch_size}")
    
    def _load_model(self, model_path: str, model_type: str) -> nn.Module:
        """Load model from checkpoint."""
        if model_type == 'resnet':
            from models.dsh_resnet import DSHResNet
            model = DSHResNet()
        elif model_type == 'resnet_lite':
            from models.dsh_resnet import DSHResNetLite
            model = DSHResNetLite()
        else:
            from models.dsh_cnn import DSHDetectorCNN
            model = DSHDetectorCNN()
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    def _window_generator(
        self,
        image: np.ndarray,
        window_size: int,
        stride: int
    ) -> Generator[Tuple[int, int, np.ndarray], None, None]:
        """Memory-safe window generator."""
        h, w = image.shape
        
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                window = image[y:y + window_size, x:x + window_size]
                if np.count_nonzero(window) >= self.min_nonzero_pixels:
                    yield x, y, window
    
    def process(
        self,
        image: np.ndarray,
        image_id: str = "survey",
        window_sizes: List[int] = [64, 128, 256],
        final_threshold: float = 0.45,
        investigate_partials: bool = True,
        output_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Run complete pipeline on an image.
        
        Args:
            image: Survey image (2D numpy array)
            image_id: Identifier for this image
            window_sizes: List of window sizes for multi-scale scanning
            final_threshold: Final detection threshold
            investigate_partials: Whether to investigate partial halos
            output_dir: Directory to save results (None = don't save)
            verbose: Print progress
            
        Returns:
            Complete results dictionary
        """
        # Ensure float32
        image = np.nan_to_num(image.astype(np.float32), nan=0.0)
        
        results = {
            'image_id': image_id,
            'image_shape': list(image.shape),
            'parameters': {
                'window_sizes': window_sizes,
                'detection_threshold': self.detection_threshold,
                'final_threshold': final_threshold,
                'stride': self.stride
            },
            'stages': {},
            'detections': [],
            'clusters': [],
            'candidates': [],
            'final_detections': [],
            'summary': {}
        }
        
        # ====== STAGE 1: Multi-scale Sliding Window ======
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 1: Multi-Scale Sliding Window Scan")
            print("=" * 60)
        
        all_detections = []
        for ws in window_sizes:
            stride = ws // 2
            dets = self._scan_at_scale(image, ws, stride, verbose)
            all_detections.extend(dets)
            
            if verbose:
                print(f"  {ws}x{ws}: {len(dets)} detections")
        
        results['stages']['stage1_total'] = len(all_detections)
        
        if verbose:
            print(f"\nStage 1 complete: {len(all_detections)} total detections")
        
        # ====== STAGE 2: Arc Direction Analysis ======
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 2: Arc Direction Analysis")
            print("=" * 60)
        
        for det in all_detections:
            window = image[det.y:det.y+det.window_size, det.x:det.x+det.window_size]
            arc_dir, edge_scores, needs_inv = self.arc_analyzer.analyze(window)
            
            det.arc_direction = arc_dir
            det.edge_scores = edge_scores
            det.needs_investigation = needs_inv
        
        partial_count = sum(1 for d in all_detections if d.needs_investigation)
        results['stages']['stage2_partials'] = partial_count
        
        if verbose:
            # Arc distribution
            arc_dist = {}
            for d in all_detections:
                arc_dist[d.arc_direction.value] = arc_dist.get(d.arc_direction.value, 0) + 1
            print(f"Arc distribution: {arc_dist}")
            print(f"Partials needing investigation: {partial_count}")
        
        # ====== STAGE 3: Spatial Clustering ======
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 3: Spatial Clustering")
            print("=" * 60)
        
        clusters = self.clusterer.cluster(all_detections)
        
        # Assign cluster IDs
        for cluster_id, indices in enumerate(clusters):
            for idx in indices:
                all_detections[idx].cluster_id = cluster_id
        
        results['stages']['stage3_clusters'] = len(clusters)
        results['clusters'] = [
            {'cluster_id': i, 'size': len(c), 'detection_indices': c}
            for i, c in enumerate(clusters)
        ]
        
        if verbose:
            print(f"Found {len(clusters)} clusters")
            sizes = [len(c) for c in clusters]
            if sizes:
                print(f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
        
        # ====== STAGE 4: Multi-Scale Investigation ======
        if investigate_partials and partial_count > 0:
            if verbose:
                print("\n" + "=" * 60)
                print("STAGE 4: Multi-Scale Investigation")
                print("=" * 60)
            
            to_investigate = [d for d in all_detections if d.needs_investigation]
            to_investigate.sort(key=lambda d: -d.probability)
            
            for i, det in enumerate(to_investigate):
                ms_results = self.investigator.investigate(image, det)
                det.multi_scale_results = ms_results
                
                if verbose and (i + 1) % 50 == 0:
                    print(f"  Investigated {i + 1}/{len(to_investigate)}")
            
            results['stages']['stage4_investigated'] = len(to_investigate)
            
            if verbose:
                print(f"Investigated {len(to_investigate)} partial detections")
        
        # ====== STAGE 5: Large Halo Inference ======
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 5: Large Halo Inference")
            print("=" * 60)
        
        candidates = []
        for cluster_indices in clusters:
            candidate = self.clusterer.analyze_cluster(all_detections, cluster_indices)
            candidate.candidate_id = len(candidates)
            candidates.append(candidate)
        
        results['candidates'] = [c.to_dict() for c in candidates]
        
        if verbose:
            type_dist = {}
            for c in candidates:
                type_dist[c.candidate_type.value] = type_dist.get(c.candidate_type.value, 0) + 1
            print(f"Candidate types: {type_dist}")
        
        # ====== STAGE 6: Final Filtering ======
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 6: Final Filtering & Validation")
            print("=" * 60)
        
        # Build map from detection index to its cluster's candidate
        det_idx_to_candidate = {}
        for cand in candidates:
            for det_idx in cand.contributing_detections:
                det_idx_to_candidate[det_idx] = cand
        
        # Filter detections using CLUSTER CONFIDENCE
        final_detections = []
        rejected_isolated = 0
        rejected_weak = 0
        
        for i, det in enumerate(all_detections):
            # Get candidate info
            candidate = det_idx_to_candidate.get(i)
            
            # Base probability
            base_prob = det.probability
            ms_prob = det.multi_scale_results.get('max_probability', 0)
            
            # CRITICAL: Use CLUSTER confidence to validate detections
            if candidate is not None:
                cluster_confidence = candidate.confidence_score
                
                # If cluster has low confidence, detection inherits that
                # This kills "orphan arcs" - isolated edges with no supporting neighbors
                if cluster_confidence < 0.25:
                    # Isolated noise - reject completely
                    rejected_isolated += 1
                    continue
                elif cluster_confidence < 0.5:
                    # Weak evidence - cap the detection probability
                    effective_prob = min(max(base_prob, ms_prob), 0.44)  # Below threshold
                    rejected_weak += 1
                else:
                    # Valid cluster - use max of base and multi-scale
                    effective_prob = max(base_prob, ms_prob)
            else:
                # No cluster info - use base probability
                effective_prob = max(base_prob, ms_prob)
            
            # Apply final threshold
            if effective_prob >= final_threshold:
                det.probability = effective_prob
                det.confidence = self._get_confidence(effective_prob)
                final_detections.append(det)
        
        if verbose:
            print(f"Rejected {rejected_isolated} isolated detections (orphan arcs)")
            print(f"Rejected {rejected_weak} weak cluster detections")
        
        # Apply NMS
        final_detections = self._apply_nms(final_detections)
        
        results['detections'] = [d.to_dict() for d in all_detections]
        results['final_detections'] = [d.to_dict() for d in final_detections]
        
        # Summary
        results['summary'] = {
            'total_detections': len(all_detections),
            'final_detections': len(final_detections),
            'clusters': len(clusters),
            'candidates': len(candidates),
            'high_confidence_candidates': sum(
                1 for c in candidates if c.confidence_score >= 0.7
            )
        }
        
        if verbose:
            print(f"Final detections after NMS: {len(final_detections)}")
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETE")
            print("=" * 60)
            print(f"Summary: {results['summary']}")
        
        # Save if output directory specified
        if output_dir:
            self._save_results(results, output_dir, image_id)
        
        return results
    
    @torch.no_grad()
    def _scan_at_scale(
        self,
        image: np.ndarray,
        window_size: int,
        stride: int,
        verbose: bool
    ) -> List[Detection]:
        """Scan image at a single scale."""
        detections = []
        batch_windows = []
        batch_positions = []
        processed = 0
        
        for x, y, window in self._window_generator(image, window_size, stride):
            batch_windows.append(window)
            batch_positions.append((x, y))
            
            if len(batch_windows) >= self.batch_size:
                batch_dets = self._process_batch(
                    batch_windows, batch_positions, window_size
                )
                detections.extend(batch_dets)
                processed += len(batch_windows)
                
                batch_windows = []
                batch_positions = []
        
        # Process remaining
        if batch_windows:
            batch_dets = self._process_batch(
                batch_windows, batch_positions, window_size
            )
            detections.extend(batch_dets)
            processed += len(batch_windows)
        
        return detections
    
    def _process_batch(
        self,
        windows: List[np.ndarray],
        positions: List[Tuple[int, int]],
        window_size: int
    ) -> List[Detection]:
        """Process a batch of windows."""
        # Preprocess
        batch = []
        for window in windows:
            w = np.log1p(np.clip(window, 0, None))
            w_min, w_max = w.min(), w.max()
            if w_max > w_min:
                w = (w - w_min) / (w_max - w_min)
            else:
                w = np.zeros_like(w)
            batch.append(w)
        
        batch = np.stack(batch)
        tensor = torch.from_numpy(batch.astype(np.float32)).unsqueeze(1)
        tensor = tensor.to(self.device)
        
        # Resize if needed
        if tensor.shape[-1] != 64:
            tensor = F.interpolate(tensor, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Inference
        outputs = self.model(tensor)
        probs = outputs.cpu().numpy().flatten()
        
        # Create detections
        detections = []
        for (x, y), prob in zip(positions, probs):
            if prob >= self.detection_threshold:
                detections.append(Detection(
                    x=x, y=y,
                    window_size=window_size,
                    probability=float(prob),
                    confidence=self._get_confidence(prob)
                ))
        
        return detections
    
    def _apply_nms(
        self,
        detections: List[Detection],
        iou_threshold: float = 0.3
    ) -> List[Detection]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []
        
        sorted_dets = sorted(detections, key=lambda d: -d.probability)
        
        kept = []
        while sorted_dets:
            best = sorted_dets.pop(0)
            kept.append(best)
            
            remaining = []
            for det in sorted_dets:
                iou = self._compute_iou(best, det)
                if iou < iou_threshold:
                    remaining.append(det)
            
            sorted_dets = remaining
        
        return kept
    
    def _compute_iou(self, d1: Detection, d2: Detection) -> float:
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
        
        return intersection / union if union > 0 else 0.0
    
    def _get_confidence(self, probability: float) -> str:
        """Get confidence category."""
        if probability >= 0.85:
            return 'DEFINITE'
        elif probability >= 0.65:
            return 'PROBABLE'
        elif probability >= 0.45:
            return 'POSSIBLE'
        elif probability >= 0.25:
            return 'UNLIKELY'
        else:
            return 'NO_HALO'
    
    def _save_results(self, results: Dict, output_dir: str, image_id: str):
        """Save results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filepath = os.path.join(output_dir, f"{image_id}_results_{timestamp}.json")
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def process_fits(
        self,
        fits_path: str,
        **kwargs
    ) -> Dict[str, any]:
        """
        Process a FITS file.
        
        Uses memory mapping for large files.
        """
        from astropy.io import fits
        
        print(f"Loading: {fits_path}")
        
        with fits.open(fits_path, memmap=True) as hdul:
            image = hdul[0].data
            if image is None and len(hdul) > 1:
                image = hdul[1].data
            
            image = image.astype(np.float32)
            
            # Extract image_id from filename
            image_id = os.path.splitext(os.path.basename(fits_path))[0]
            
            return self.process(image, image_id=image_id, **kwargs)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def quick_scan(
    fits_path: str,
    model_path: str,
    model_type: str = 'resnet',
    output_dir: Optional[str] = './results'
) -> Dict:
    """
    Quick scan of a FITS file.
    
    Example:
        results = quick_scan('survey.fits', 'model.pth')
    """
    pipeline = DSHPipeline(model_path, model_type)
    return pipeline.process_fits(fits_path, output_dir=output_dir)


if __name__ == "__main__":
    print("DSH Pipeline - Complete Detection System")
    print("=" * 60)
    print("""
Stages:
  1. Multi-scale sliding window scan
  2. Arc direction analysis
  3. Spatial clustering
  4. Multi-scale investigation
  5. Large halo inference
  6. Export results

Usage:
    from dsh_pipeline import DSHPipeline, quick_scan
    
    # Full control
    pipeline = DSHPipeline(
        model_path='checkpoints/best_model.pth',
        model_type='resnet'
    )
    results = pipeline.process(image)
    
    # Quick scan
    results = quick_scan('survey.fits', 'model.pth')
    """)