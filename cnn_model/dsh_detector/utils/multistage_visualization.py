"""
Multi-Stage Pipeline Visualization
===================================
Visualization tools for the multi-stage DSH detection pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Optional, Tuple
import json


# Color scheme for detection types
DETECTION_COLORS = {
    'full_halo': '#00FF00',       # Green
    'partial_top': '#FF6B6B',     # Red
    'partial_bottom': '#FF6B6B',  # Red
    'partial_left': '#FFB347',    # Orange
    'partial_right': '#FFB347',   # Orange
    'partial_corner': '#FF0000',  # Bright Red
    'uncertain': '#FFFF00'        # Yellow
}

CATEGORY_COLORS = {
    'DEFINITE_HALO': '#00FF00',
    'PROBABLE_HALO': '#90EE90',
    'POSSIBLE_HALO': '#FFFF00',
    'UNLIKELY_HALO': '#FFA500',
    'NO_HALO': '#808080'
}


def plot_survey_with_detections(
    survey_image: np.ndarray,
    results: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (16, 12),
    color_by: str = 'detection_type'  # or 'category'
):
    """
    Plot survey image with all detection stages visualized.
    
    Args:
        survey_image: The survey image array
        results: Results dictionary from MultiStagePipeline
        save_path: Optional path to save figure
        show: Whether to display
        figsize: Figure size
        color_by: Color boxes by 'detection_type' or 'category'
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Log transform for display
    img_display = np.log1p(np.clip(survey_image, 0, None))
    
    # Panel 1: Full survey with all Stage 1 candidates
    ax1 = axes[0, 0]
    ax1.imshow(img_display, cmap='viridis', origin='lower')
    ax1.set_title(f"Stage 1: Initial Scan ({len(results.get('stage1_candidates', []))} candidates)")
    
    for det in results.get('stage1_candidates', []):
        rect = patches.Rectangle(
            (det['x'], det['y']), det['window_size'], det['window_size'],
            linewidth=1, edgecolor='white', facecolor='none', alpha=0.5
        )
        ax1.add_patch(rect)
    
    # Panel 2: Partial halo flags
    ax2 = axes[0, 1]
    ax2.imshow(img_display, cmap='viridis', origin='lower')
    ax2.set_title(f"Stage 2: Partial Halo Flags ({len(results.get('stage2_partial_flags', []))} flagged)")
    
    for det in results.get('stage2_partial_flags', []):
        color = DETECTION_COLORS.get(det.get('detection_type', 'uncertain'), 'yellow')
        rect = patches.Rectangle(
            (det['x'], det['y']), det['window_size'], det['window_size'],
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax2.add_patch(rect)
        
        # Add arrow indicating direction of partial
        cx = det['x'] + det['window_size'] // 2
        cy = det['y'] + det['window_size'] // 2
        
        dtype = det.get('detection_type', '')
        if 'top' in dtype:
            ax2.annotate('', xy=(cx, det['y']), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
        elif 'bottom' in dtype:
            ax2.annotate('', xy=(cx, det['y'] + det['window_size']), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
        elif 'left' in dtype:
            ax2.annotate('', xy=(det['x'], cy), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
        elif 'right' in dtype:
            ax2.annotate('', xy=(det['x'] + det['window_size'], cy), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # Panel 3: Multi-scale investigation results
    ax3 = axes[1, 0]
    ax3.imshow(img_display, cmap='viridis', origin='lower')
    ax3.set_title(f"Stage 3: Multi-Scale Investigation")
    
    for det in results.get('stage3_investigated', []):
        ms_results = det.get('multi_scale_results', {})
        best_scale = ms_results.get('best_scale', 1.0)
        max_prob = ms_results.get('max_probability', det['probability'])
        
        # Draw original window
        rect = patches.Rectangle(
            (det['x'], det['y']), det['window_size'], det['window_size'],
            linewidth=1, edgecolor='white', facecolor='none', linestyle='--'
        )
        ax3.add_patch(rect)
        
        # Draw best scale window
        cx = det['x'] + det['window_size'] // 2
        cy = det['y'] + det['window_size'] // 2
        best_size = int(det['window_size'] * best_scale)
        
        color = '#00FF00' if max_prob >= 0.65 else '#FFFF00' if max_prob >= 0.45 else '#FF6B6B'
        rect2 = patches.Rectangle(
            (cx - best_size//2, cy - best_size//2), best_size, best_size,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax3.add_patch(rect2)
        
        # Label with max probability
        ax3.text(cx, cy - best_size//2 - 5, f'{max_prob:.2f}', 
                color=color, fontsize=8, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Panel 4: Final detections
    ax4 = axes[1, 1]
    ax4.imshow(img_display, cmap='viridis', origin='lower')
    ax4.set_title(f"Final Detections ({len(results.get('final_detections', []))} confirmed)")
    
    colors = DETECTION_COLORS if color_by == 'detection_type' else CATEGORY_COLORS
    
    for det in results.get('final_detections', []):
        if color_by == 'detection_type':
            color = colors.get(det.get('detection_type', 'full_halo'), '#00FF00')
        else:
            color = colors.get(det.get('category', 'POSSIBLE_HALO'), '#FFFF00')
        
        rect = patches.Rectangle(
            (det['x'], det['y']), det['window_size'], det['window_size'],
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax4.add_patch(rect)
        
        # Add probability label
        ax4.text(det['x'] + 2, det['y'] + det['window_size'] - 5,
                f"{det['probability']:.2f}",
                color='white', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    # Add legend for detection types
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor=color, label=dtype.replace('_', ' ').title(), linewidth=2)
        for dtype, color in DETECTION_COLORS.items()
    ]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_partial_halo_detail(
    survey_image: np.ndarray,
    detection: Dict,
    context_size: int = 128,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Detailed view of a partial halo detection with edge analysis.
    
    Args:
        survey_image: Full survey image
        detection: Single detection dictionary
        context_size: Size of context region to show
        save_path: Optional save path
        show: Whether to display
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    x, y = detection['x'], detection['y']
    ws = detection['window_size']
    
    # Calculate context region
    h, w = survey_image.shape
    cx, cy = x + ws // 2, y + ws // 2
    
    x1 = max(0, cx - context_size // 2)
    y1 = max(0, cy - context_size // 2)
    x2 = min(w, cx + context_size // 2)
    y2 = min(h, cy + context_size // 2)
    
    context_region = survey_image[y1:y2, x1:x2]
    detection_window = survey_image[y:y+ws, x:x+ws]
    
    # Log transform
    context_display = np.log1p(np.clip(context_region, 0, None))
    window_display = np.log1p(np.clip(detection_window, 0, None))
    
    # Panel 1: Context region
    ax1 = axes[0, 0]
    ax1.imshow(context_display, cmap='viridis', origin='lower')
    ax1.set_title(f"Context Region ({context_size}x{context_size})")
    
    # Draw detection box in context
    rect = patches.Rectangle(
        (x - x1, y - y1), ws, ws,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax1.add_patch(rect)
    
    # Panel 2: Detection window
    ax2 = axes[0, 1]
    ax2.imshow(window_display, cmap='viridis', origin='lower')
    ax2.set_title(f"Detection Window (64x64)\nProb: {detection['probability']:.3f}")
    
    # Panel 3: Edge intensity visualization
    ax3 = axes[0, 2]
    edge_scores = detection.get('edge_scores', {})
    
    # Create edge visualization
    edge_viz = np.zeros((ws, ws))
    m = 8  # margin
    
    edge_viz[:m, :] = edge_scores.get('top', 0)
    edge_viz[-m:, :] = edge_scores.get('bottom', 0)
    edge_viz[:, :m] = edge_scores.get('left', 0)
    edge_viz[:, -m:] = edge_scores.get('right', 0)
    edge_viz[m:-m, m:-m] = edge_scores.get('center', 0)
    
    im = ax3.imshow(edge_viz, cmap='hot', origin='lower')
    ax3.set_title("Edge Intensity Analysis")
    plt.colorbar(im, ax=ax3)
    
    # Panel 4: Edge ratios bar chart
    ax4 = axes[1, 0]
    ratios = {
        'Top': edge_scores.get('top_ratio', 0),
        'Bottom': edge_scores.get('bottom_ratio', 0),
        'Left': edge_scores.get('left_ratio', 0),
        'Right': edge_scores.get('right_ratio', 0)
    }
    
    colors = ['red' if v > 0.7 else 'green' for v in ratios.values()]
    bars = ax4.bar(ratios.keys(), ratios.values(), color=colors)
    ax4.axhline(y=0.7, color='orange', linestyle='--', label='Threshold')
    ax4.set_ylabel('Edge/Center Ratio')
    ax4.set_title('Edge Analysis\n(Red = Possible Partial Halo)')
    ax4.legend()
    
    # Panel 5: Multi-scale results
    ax5 = axes[1, 1]
    ms_results = detection.get('multi_scale_results', {})
    scales_data = ms_results.get('scales', {})
    
    if scales_data:
        scale_names = list(scales_data.keys())
        scale_probs = [scales_data[s]['probability'] for s in scale_names]
        
        ax5.bar(scale_names, scale_probs, color='steelblue')
        ax5.axhline(y=0.65, color='green', linestyle='--', label='Probable threshold')
        ax5.axhline(y=0.45, color='orange', linestyle='--', label='Possible threshold')
        ax5.set_ylabel('Probability')
        ax5.set_title(f"Multi-Scale Analysis\nBest: {ms_results.get('best_scale', 'N/A')}")
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'No multi-scale\ndata available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Multi-Scale Analysis')
    
    # Panel 6: Detection info
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    info_text = f"""
    Detection Information
    =====================
    Position: ({x}, {y})
    Window Size: {ws}x{ws}
    Probability: {detection['probability']:.4f}
    Category: {detection.get('category', 'N/A')}
    
    Detection Type: {detection.get('detection_type', 'N/A')}
    Needs Investigation: {detection.get('needs_investigation', False)}
    
    Multi-Scale Results:
    - Max Probability: {ms_results.get('max_probability', 'N/A')}
    - Best Scale: {ms_results.get('best_scale', 'N/A')}
    - Recommendation: {ms_results.get('recommended_action', 'N/A')}
    """
    
    ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f"Partial Halo Analysis - Detection at ({x}, {y})", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def create_detection_report(
    results: Dict,
    output_path: str = 'detection_report.txt'
):
    """
    Create a text report of detection results.
    
    Args:
        results: Results dictionary from MultiStagePipeline
        output_path: Path to save report
    """
    report = []
    report.append("=" * 70)
    report.append("DSH DETECTION REPORT - MULTI-STAGE PIPELINE")
    report.append("=" * 70)
    
    # Survey info
    report.append(f"\nSurvey Shape: {results.get('survey_shape', 'N/A')}")
    
    # Parameters
    params = results.get('parameters', {})
    report.append(f"\nParameters:")
    report.append(f"  Window Size: {params.get('window_size', 'N/A')}")
    report.append(f"  Stride: {params.get('stride', 'N/A')}")
    report.append(f"  Initial Threshold: {params.get('initial_threshold', 'N/A')}")
    report.append(f"  Final Threshold: {params.get('final_threshold', 'N/A')}")
    
    # Statistics
    stats = results.get('statistics', {})
    report.append(f"\nStatistics:")
    report.append(f"  Total Windows Scanned: {stats.get('total_windows_scanned', 'N/A')}")
    report.append(f"  Stage 1 Candidates: {stats.get('stage1_candidates', 'N/A')}")
    report.append(f"  Stage 2 Partial Flags: {stats.get('stage2_partial_flags', 'N/A')}")
    report.append(f"  Final Detections: {stats.get('final_detections', 'N/A')}")
    
    # Detection types
    report.append(f"\nDetection Types:")
    for dtype, count in stats.get('detection_types', {}).items():
        report.append(f"  {dtype}: {count}")
    
    # Final detections detail
    report.append(f"\n{'=' * 70}")
    report.append("FINAL DETECTIONS")
    report.append("=" * 70)
    
    for i, det in enumerate(results.get('final_detections', [])):
        report.append(f"\nDetection {i + 1}:")
        report.append(f"  Position: ({det['x']}, {det['y']})")
        report.append(f"  Probability: {det['probability']:.4f}")
        report.append(f"  Category: {det['category']}")
        report.append(f"  Type: {det.get('detection_type', 'N/A')}")
        
        ms = det.get('multi_scale_results', {})
        if ms:
            report.append(f"  Multi-Scale Max Prob: {ms.get('max_probability', 'N/A')}")
            report.append(f"  Recommended Action: {ms.get('recommended_action', 'N/A')}")
    
    # Write report
    report_text = '\n'.join(report)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"Report saved to {output_path}")
    return report_text


if __name__ == "__main__":
    print("Multi-Stage Pipeline Visualization Tools")
    print("=" * 50)
    print("""
Available functions:
    - plot_survey_with_detections(survey_image, results)
    - plot_partial_halo_detail(survey_image, detection)
    - create_detection_report(results, output_path)
    """)
