# DSH Detector - Dust Scattering Halo Detection CNN

A PyTorch-based convolutional neural network for detecting dust scattering halos (DSH) in X-ray astronomical images.

## Project Overview

This module is **Part 5 (Model Selection & Training)** of the DSH detection project, which aims to:
1. Train a CNN to distinguish X-ray images containing dust scattering halos from those without
2. Provide confidence scores and categorization for detections
3. Enable scanning of large survey images (e.g., eROSITA) using sliding window detection

## Directory Structure

```
dsh_detector/
├── models/
│   ├── __init__.py
│   └── dsh_cnn.py          # CNN architecture (Full and Lite versions)
├── data/
│   ├── __init__.py
│   └── dataset.py          # PyTorch Dataset for FITS loading
├── utils/
│   ├── __init__.py
│   └── visualization.py    # Plotting utilities
├── train.py                # Training script
├── inference.py            # Inference and survey scanning
├── run_training.sh         # Bash script for easy training
└── README.md              # This file
```

## Installation

### Requirements

```bash
# Create conda environment (recommended)
conda create -n dsh python=3.10
conda activate dsh

# Install PyTorch (adjust for your CUDA version)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision

# Install other dependencies
pip install numpy pandas matplotlib astropy scipy
```

### Quick Install
```bash
pip install torch torchvision numpy pandas matplotlib astropy scipy
```

## Usage

### 1. Training the Model

#### Option A: Using the bash script
```bash
# Edit run_training.sh to set your paths
nano run_training.sh

# Run training
bash run_training.sh
```

#### Option B: Direct Python command
```bash
python train.py \
    --csv_path /path/to/balanced_10k_vND7_with_split.csv \
    --data_root /data3/efeoztaban/vND7_directories_shrunk_clouds/ \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --model full \
    --checkpoint_dir ./checkpoints
```

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv_path` | Required | Path to CSV with image metadata |
| `--data_root` | Required | Root directory with FITS files |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 0.001 | Initial learning rate |
| `--model` | full | Model type: `full` or `lite` |
| `--negative_ratio` | 1.0 | Ratio of negative to positive samples |
| `--dropout` | 0.3 | Dropout rate |
| `--optimizer` | adam | Optimizer: `adam`, `adamw`, `sgd` |
| `--scheduler` | plateau | LR scheduler: `plateau`, `cosine` |
| `--patience` | 15 | Early stopping patience |
| `--use_amp` | False | Use mixed precision training |

### 2. Single Image Inference

```python
from inference import DSHInference

# Load model
detector = DSHInference(
    model_path='checkpoints/best_model.pth',
    model_type='full'
)

# Predict on a single image
probability, category = detector.predict('path/to/image.fits')
print(f"Probability: {probability:.4f}")
print(f"Category: {category}")
```

Or via command line:
```bash
python inference.py \
    --model checkpoints/best_model.pth \
    --image path/to/image.fits
```

### 3. Survey Scanning (Sliding Window)

```python
from inference import SurveyScanner

# Initialize scanner
scanner = SurveyScanner(
    model_path='checkpoints/best_model.pth',
    model_type='full'
)

# Scan a survey image
detections = scanner.scan_survey(
    survey_path='path/to/erosita_survey.fits',
    window_size=64,
    stride=32,
    threshold=0.45,
    output_path='detections.json'
)

# Print detections
for det in detections[:10]:
    print(f"Position: ({det.x}, {det.y}), Prob: {det.probability:.3f}, Category: {det.category}")
```

Or via command line:
```bash
python inference.py \
    --model checkpoints/best_model.pth \
    --survey path/to/erosita_survey.fits \
    --window_size 64 \
    --stride 32 \
    --threshold 0.45 \
    --output detections.json
```

### 4. Visualization

```python
from utils.visualization import (
    plot_training_history,
    plot_survey_detections,
    plot_confusion_matrix
)

# Plot training curves
plot_training_history('checkpoints/training_history.json', save_path='training.png')

# Plot detections on survey
import json
from astropy.io import fits

survey = fits.open('survey.fits')[0].data
with open('detections.json') as f:
    detections = json.load(f)['detections']

plot_survey_detections(survey, detections, save_path='survey_detections.png')
```

## Model Architecture

### Full Model (DSHDetectorCNN)
- **Parameters**: ~1.2M
- **Input**: 64×64 grayscale image
- **Architecture**: 
  - 4 convolutional blocks with batch normalization
  - Global average pooling
  - 3 fully connected layers with dropout
  - Sigmoid output for probability

### Lite Model (DSHDetectorCNNLite)
- **Parameters**: ~200K
- **Use case**: Faster inference for survey scanning
- **Architecture**: Simplified version with fewer channels

## Confidence Categories

The model outputs a probability [0, 1] which is categorized as:

| Probability | Category |
|-------------|----------|
| ≥ 0.85 | DEFINITE_HALO |
| 0.65 - 0.85 | PROBABLE_HALO |
| 0.45 - 0.65 | POSSIBLE_HALO |
| 0.25 - 0.45 | UNLIKELY_HALO |
| < 0.25 | NO_HALO |

## Training Data

The training data consists of:
- **Positive samples**: Synthetic DSH images from FITS files
- **Negative samples**: Black/zero images (generated during training)

The dataset is loaded from a CSV file with the following columns:
- `distance`: Source distance
- `nh_unif`: Uniform hydrogen column density
- `nh_wco`: Molecular cloud hydrogen column density
- `relative_path`: Path to FITS file
- `split`: train/val/test split

## Expected Performance

With the default settings on the synthetic dataset:
- **Training Accuracy**: ~95-99%
- **Validation Accuracy**: ~93-97%
- **Test Accuracy**: ~92-96%

Note: Performance on real eROSITA data will depend on the SIXTE processing and domain adaptation.

## Future Improvements

After running data through SIXTE (Part 4), consider:
1. Fine-tuning on SIXTE-processed images
2. Adding more realistic negative samples (PSF-only images)
3. Implementing attention mechanisms for better localization
4. Adding multi-scale detection for halos of different sizes

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size: `--batch_size 16`
   - Use lite model: `--model lite`
   - Enable AMP: `--use_amp`

2. **Slow Training**
   - Increase workers: `--num_workers 8`
   - Use GPU: Ensure CUDA is available

3. **Poor Accuracy**
   - Check data paths are correct
   - Verify FITS files are loading properly
   - Try different learning rates

### Checking GPU Availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Contact

Part of the DSH Detection Project
- Part 1: X-ray detector basics
- Part 2: Dust scattering halo basics
- Part 3: Synthetic data curation
- Part 4: Instrument realism with SIXTE
- **Part 5: Model selection and training** (this module)
- Part 6: Testing & eROSITA

## License

Internal use for DSH Detection Project
