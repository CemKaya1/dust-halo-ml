#!/bin/bash
# ============================================================
# DSH Detector - Training Script
# ============================================================
# 
# This script runs the complete training pipeline for the
# Dust Scattering Halo detection CNN.
#
# Usage:
#   bash run_training.sh
#
# Make sure to update the paths below before running!
# ============================================================

# ==================== CONFIGURATION ====================
# UPDATE THESE PATHS FOR YOUR SERVER

# Path to the CSV file with image metadata
CSV_PATH="/home/pure26/cnn_model/dsh_detector/balanced_10k_vND7_with_split.csv"

# Root directory containing FITS files
DATA_ROOT="/data3/efeoztaban/vND7_directories_shrunk_clouds/"

# Output directory for checkpoints
CHECKPOINT_DIR="./checkpoints/run_$(date +%Y%m%d_%H%M%S)"

# ==================== TRAINING PARAMETERS ====================
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.001
MODEL="full"  # Options: full, lite
NEGATIVE_RATIO=1.0  # 1.0 means equal positives and negatives

# ==================== RUN TRAINING ====================
echo "============================================================"
echo "DSH Detector Training"
echo "============================================================"
echo "CSV Path: $CSV_PATH"
echo "Data Root: $DATA_ROOT"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "============================================================"

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Run training
python train.py \
    --csv_path "$CSV_PATH" \
    --data_root "$DATA_ROOT" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --model $MODEL \
    --negative_ratio $NEGATIVE_RATIO \
    --optimizer adamw \
    --scheduler plateau \
    --patience 15 \
    --num_workers 4

echo "============================================================"
echo "Training complete!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "============================================================"
