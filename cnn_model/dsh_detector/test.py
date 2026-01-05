"""
Test DSH Pipeline on eROSITA Data
==================================
Tests the trained model on real eROSITA survey data.

Usage:
    CUDA_VISIBLE_DEVICES="" python3 test_final.py

"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits

from dsh_pipeline import DSHPipeline


def main():
    print("=" * 60)
    print("DSH PIPELINE TEST ON eROSITA")
    print("=" * 60)
    
    # ===== CONFIGURATION =====
    MODEL_PATH = './checkpoints_final/best_model.pth'
    EROSITA_PATH = '/data3/pure26/eRASS/test_data/150/203/EXP_010/em01_203150_024_Image_c010.fits.gz'
    OUTPUT_DIR = './test_results'
    
    # ===== LOAD MODEL & CREATE PIPELINE =====
    print("\nInitializing pipeline...")
    pipeline = DSHPipeline(
        model_path=MODEL_PATH,
        model_type='resnet',
        window_size=64,
        stride=32,
        batch_size=128,
        detection_threshold=0.35,
        cluster_proximity=100
    )
    
    # ===== LOAD eROSITA IMAGE =====
    print(f"\nLoading: {EROSITA_PATH}")
    with fits.open(EROSITA_PATH, memmap=True) as hdul:
        image = hdul[0].data.astype(np.float32)
    
    print(f"Image shape: {image.shape}")
    print(f"Value range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"Non-zero pixels: {np.count_nonzero(image)} ({100*np.count_nonzero(image)/image.size:.2f}%)")
    
    # ===== RUN PIPELINE =====
    print("\n" + "=" * 60)
    print("RUNNING PIPELINE")
    print("=" * 60)
    
    results = pipeline.process(
        image,
        image_id='erosita_150_203_024',
        window_sizes=[64, 128, 256],
        final_threshold=0.45,
        investigate_partials=True,
        output_dir=OUTPUT_DIR,
        verbose=True
    )
    
    # ===== PRINT SUMMARY =====
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    summary = results['summary']
    print(f"Total detections: {summary['total_detections']}")
    print(f"Final detections (after NMS): {summary['final_detections']}")
    print(f"Clusters found: {summary['clusters']}")
    print(f"Halo candidates: {summary['candidates']}")
    print(f"High confidence candidates: {summary['high_confidence_candidates']}")
    
    # Probability distribution
    if results['detections']:
        probs = [d['probability'] for d in results['detections']]
        print(f"\nProbability distribution:")
        print(f"  Min: {min(probs):.4f}")
        print(f"  Max: {max(probs):.4f}")
        print(f"  Mean: {np.mean(probs):.4f}")
        print(f"  Std: {np.std(probs):.4f}")
        
        # By confidence
        confidence_counts = {}
        for d in results['detections']:
            conf = d['confidence']
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        print(f"\nBy confidence level:")
        for conf in ['DEFINITE', 'PROBABLE', 'POSSIBLE', 'UNLIKELY', 'NO_HALO']:
            count = confidence_counts.get(conf, 0)
            if count > 0:
                print(f"  {conf}: {count}")
    
    # Arc distribution
    if results['detections']:
        arc_counts = {}
        for d in results['detections']:
            arc = d['arc_direction']
            arc_counts[arc] = arc_counts.get(arc, 0) + 1
        
        print(f"\nArc direction distribution:")
        for arc, count in sorted(arc_counts.items(), key=lambda x: -x[1]):
            print(f"  {arc}: {count}")
    
    # Candidates
    if results['candidates']:
        print(f"\nHalo candidates:")
        for cand in results['candidates'][:10]:  # Show top 10
            print(f"  ID {cand['candidate_id']}: {cand['candidate_type']}, "
                  f"pos=({cand['center_x']:.0f}, {cand['center_y']:.0f}), "
                  f"radius={cand['estimated_radius']:.0f}, "
                  f"confidence={cand['confidence_score']:.2f}")
    
    # ===== VISUALIZATION =====
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 1. Full image
    ax1 = axes[0, 0]
    ax1.imshow(np.log1p(image), cmap='viridis', origin='lower')
    ax1.set_title(f"eROSITA Image\n{image.shape[0]}x{image.shape[1]} pixels")
    
    # 2. Detections
    ax2 = axes[0, 1]
    ax2.imshow(np.log1p(image), cmap='viridis', origin='lower')
    
    colors = {
        'DEFINITE': 'red',
        'PROBABLE': 'orange', 
        'POSSIBLE': 'yellow',
        'UNLIKELY': 'cyan'
    }
    
    # Show final detections
    for det in results['final_detections'][:100]:
        color = colors.get(det['confidence'], 'white')
        rect = patches.Rectangle(
            (det['x'], det['y']), det['window_size'], det['window_size'],
            linewidth=1, edgecolor=color, facecolor='none', alpha=0.8
        )
        ax2.add_patch(rect)
    
    ax2.set_title(f"Final Detections\n{len(results['final_detections'])} total (showing top 100)")
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor=c, label=l, linewidth=2)
        for l, c in colors.items()
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # 3. Probability histogram
    ax3 = axes[1, 0]
    if results['detections']:
        probs = [d['probability'] for d in results['detections']]
        ax3.hist(probs, bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(0.45, color='orange', linestyle='--', label='Threshold (0.45)')
        ax3.axvline(0.65, color='red', linestyle='--', label='Probable (0.65)')
        ax3.set_xlabel('Probability')
        ax3.set_ylabel('Count')
        ax3.set_title('Detection Probability Distribution')
        ax3.legend()
        ax3.set_yscale('log')
    else:
        ax3.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Halo candidates
    ax4 = axes[1, 1]
    ax4.imshow(np.log1p(image), cmap='viridis', origin='lower')
    
    type_colors = {
        'full_halo': 'lime',
        'large_halo_inferred': 'red',
        'edge_candidate': 'yellow',
        'scattered_arcs': 'orange',
        'uncertain': 'cyan'
    }
    
    for cand in results['candidates']:
        color = type_colors.get(cand['candidate_type'], 'white')
        circle = plt.Circle(
            (cand['center_x'], cand['center_y']),
            cand['estimated_radius'],
            fill=False,
            edgecolor=color,
            linewidth=2,
            alpha=0.8
        )
        ax4.add_patch(circle)
        ax4.plot(cand['center_x'], cand['center_y'], 'x', color=color, markersize=8)
    
    ax4.set_title(f"Halo Candidates\n{len(results['candidates'])} total")
    
    # Legend
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor=c, label=t.replace('_', ' ').title(), linewidth=2)
        for t, c in type_colors.items()
    ]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/erosita_results.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR}/erosita_results.png")
    
    # ===== COMPARISON WITH PREVIOUS MODELS =====
    print("\n" + "=" * 60)
    print("COMPARISON WITH PREVIOUS MODELS")
    print("=" * 60)
    print("Model 1 (black negatives): 12,572 DEFINITE (everything = halo)")
    print("Model 2 (improved negatives): 0 DEFINITE (nothing = halo)")
    print(f"Model 3 (ResNet + v2 dataset): {confidence_counts.get('DEFINITE', 0)} DEFINITE")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("Files:")
    print(f"  - erosita_results.png (visualization)")
    print(f"  - erosita_150_203_024_results_*.json (full results)")


if __name__ == "__main__":
    main()