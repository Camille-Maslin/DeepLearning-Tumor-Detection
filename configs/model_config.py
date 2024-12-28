"""
Configuration file for the Brain Tumor Classification model.

This module contains all the configuration parameters used in the model training
and evaluation process.
"""

import os

# Model parameters
BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 20

# Training parameters
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.1
SHUFFLE_BUFFER_SIZE = 1000

# Class names
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
N_CLASSES = len(CLASS_NAMES)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "base_data")
TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "testing_images")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "src", "models", "brain_tumor_cnn", "brain_tumor_model.keras")
RESULTS_DIR = os.path.join(BASE_DIR, "src", "models", "brain_tumor_cnn", "results") 