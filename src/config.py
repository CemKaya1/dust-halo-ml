import os

# Paths
DATA_ROOT = "/path/to/your/images/"  # Change this to your actual root directory
METADATA_PATH = "data/curated_set.csv"
MODEL_SAVE_PATH = "models/"

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
IMAGE_SIZE = (50, 50) # Based on your PDF

# Data Specs
# If you want to classify Halo vs Background, set this to 1. 
# If you want to detect distance (Regression), set to 1.
# If you strictly want Halo (1) vs Empty (0), you need empty images.
NUM_CLASSES = 1