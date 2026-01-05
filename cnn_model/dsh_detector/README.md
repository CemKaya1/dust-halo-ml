# DSH Detector - Dust Scattering Halo Detection Pipeline

A CNN-based detection pipeline for identifying Dust Scattering Halos (DSH) in X-ray astronomical images from eROSITA.

## Project Overview

This project implements a multi-stage detection pipeline that:
1. Uses a ResNet CNN to identify potential DSH candidates
2. Analyzes arc geometry and spatial coherence
3. Clusters detections and validates using physical constraints
4. Filters false positives (orphan arcs, noise artifacts)

## File Structure

```
dsh_detector/
│
├── models/                          # Neural network architectures
│   ├── __init__.py                  # Package init (can be empty)
│   ├── dsh_resnet.py                # ResNet with LeakyReLU, strided conv
│   └── dsh_resnet_attention.py      # ResNet with attention mechanism (optional)
│
├── data/                            # Dataset and data loading
│   ├── __init__.py                  # Package init (can be empty)
│   └── dataset_v2.py                # Dataset with noise injection + hard negatives
│
├── checkpoints_final/               # Saved model weights (created during training)
│   └── best_model.pth
│
├── test_results/                    # Test outputs (created during testing)
│   ├── *_results.png                # Visualizations
│   ├── survey_summary_*.json        # Survey statistics
│   └── strong_candidates_*.json     # Detected halo candidates
│
├── dsh_pipeline_final.py            # Main detection pipeline
├── train_final.py                   # Training script
├── test_final.py                    # Testing script (single tile + full survey)
│
├── balanced_10k_vND7_with_split.csv # Training data CSV
└── README.md                        # This file
```

## Quick Start

### 1. Training

```bash
cd /home/pure26/cnn_model/dsh_detector

# Train on GPU (recommended):
python3 train_final.py

# Train on CPU (slower):
CUDA_VISIBLE_DEVICES="" python3 train_final.py
```

**Training output:**
- Model saved to: `./checkpoints_final/best_model.pth`
- Training history: `./checkpoints_final/history.json`

### 2. Testing

```bash
# Quick test on single tile:
python3 test_final.py

# Full survey - scan ALL eROSITA tiles:
python3 test_final.py --full-survey

# Test specific tile:
python3 test_final.py --tile /path/to/file.fits

# Custom output directory:
python3 test_final.py --full-survey --output ./my_results
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DSH DETECTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STAGE 1: Multi-Scale CNN Scan                                  │
│  ├── Window sizes: 64x64, 128x128, 256x256                      │
│  ├── Sliding window with stride = window_size / 2               │
│  └── Initial detection threshold: 0.35                          │
│                                                                 │
│  STAGE 2: Arc Geometry Analysis                                 │
│  ├── 9-region intensity analysis                                │
│  ├── Determine arc direction (TOP, BOTTOM, LEFT, RIGHT, etc.)   │
│  └── Calculate arc quality score                                │
│                                                                 │
│  STAGE 3: Spatial Clustering                                    │
│  ├── Union-Find algorithm with spatial bucketing                │
│  ├── Proximity threshold: 100 pixels                            │
│  └── Group nearby detections into clusters                      │
│                                                                 │
│  STAGE 4: Multi-Scale Investigation                             │
│  ├── Zoom out on partial detections (0.5x, 1.0x, 1.5x, 2.0x)    │
│  ├── Reflection padding for edge candidates                     │
│  └── Find best scale for each detection                         │
│                                                                 │
│  STAGE 5: Physical Coherence Analysis                           │
│  ├── Angular coverage (0-360°)                                  │
│  ├── Radial consistency                                         │
│  ├── Center agreement                                           │
│  └── Classify: STRONG_HALO, PROBABLE_HALO, ARTIFACT, etc.       │
│                                                                 │
│  STAGE 6: Final Filtering                                       │
│  ├── Evidence-based scoring (60% CNN + 40% geometry)            │
│  ├── Reject artifacts and isolated detections                   │
│  ├── Apply Non-Maximum Suppression (NMS)                        │
│  └── Final threshold: 0.45                                      │  
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Output Classification

| Category | Confidence | Description |
|----------|------------|-------------|
| `STRONG_HALO` | 0.85-0.95 | High confidence, good geometry |
| `PROBABLE_HALO` | 0.70-0.85 | Good evidence, minor issues |
| `PARTIAL_HALO` | 0.50-0.70 | Incomplete but consistent |
| `WEAK_CANDIDATE` | 0.35-0.50 | Low evidence, needs follow-up |
| `ARC_EMISSION` | 0.30-0.40 | Arc-like but not full halo |
| `ARTIFACT` | < 0.25 | Rejected as noise/artifact |

## Key Features

### Model Architecture (ResNet)
- **Skip connections**: Preserve sharp PSF vs fuzzy halo distinction
- **LeakyReLU**: Keeps faint signals alive in sparse X-ray data
- **Strided convolutions**: No MaxPool (doesn't delete low-intensity data)
- **Dropout 0.3-0.4**: Regularization for small dataset

### Dataset Improvements (v2)
- **Noise injection**: Adds Poisson noise to synthetic halos
- **Hard negatives**: Bright PSF without halo ring (30% of negatives)
- **Real eROSITA backgrounds**: Mixed into negative samples
- **Balanced distribution**: Multiple negative types

### Geometry Validation
- **Isolation penalty**: Single edge detection = likely noise (confidence 0.1)
- **Spread requirement**: Large halos need arcs on opposite sides
- **Center agreement**: Arc directions must point to common center

## Configuration

Edit `test_final.py` CONFIG section to change:

```python
CONFIG = {
    'model_path': './checkpoints_final/best_model.pth',
    'model_type': 'resnet',
    'erosita_base': '/data3/pure26/eRASS/test_data',
    'default_tile': '/data3/pure26/eRASS/test_data/150/203/EXP_010/em01_203150_024_Image_c010.fits.gz',
    'output_dir': './test_results',
    'window_sizes': [64, 128, 256],
    'detection_threshold': 0.35,
    'final_threshold': 0.45
}
```

## Troubleshooting

### Model not found
```bash
# Train the model first:
python3 train_final.py
```

### Corrupted checkpoint
```bash
# Delete and retrain:
rm -rf ./checkpoints_final/
python3 train_final.py
```

### Import errors
```bash
# Empty the __init__.py files:
echo "" > models/__init__.py
echo "" > data/__init__.py
```

### GPU not detected
```bash
# Check CUDA:
python3 -c "import torch; print(torch.cuda.is_available())"

# Force CPU:
CUDA_VISIBLE_DEVICES="" python3 train_final.py
```

## Results Interpretation

After running `python3 test_final.py --full-survey`:

1. **`survey_summary_*.json`**: Overall statistics
   - Total tiles processed
   - Detection counts by energy band
   - Confidence distribution

2. **`strong_candidates_*.json`**: High-confidence detections
   - Position (center_x, center_y)
   - Estimated radius
   - Confidence score
   - Classification reason

3. **`survey_results_*.png`**: Visualization
   - Detections by energy band
   - Confidence distribution pie chart
   - Candidate types bar chart

## Future Improvements

1. **SIXTE Integration**: Process synthetic data through SIXTE for realistic instrument effects
2. **Attention Model**: Use `dsh_resnet_attention.py` for better ring detection
3. **Cross-tile Validation**: Check edge candidates against adjacent tiles
4. **Known DSH Catalog**: Validate against confirmed DSH locations

## Author

DSH Detection Project - Part 5 (Model Selection & Training)

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Astropy
- Matplotlib
- scikit-learn (for metrics)