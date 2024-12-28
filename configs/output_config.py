"""
Output Configuration

This module defines the paths and structure for all output files,
including model artifacts, visualizations, and logs.
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Model artifacts
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Visualizations
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
TRAINING_PLOTS_DIR = os.path.join(VISUALIZATIONS_DIR, "training")
EVALUATION_PLOTS_DIR = os.path.join(VISUALIZATIONS_DIR, "evaluation")
PREDICTION_PLOTS_DIR = os.path.join(VISUALIZATIONS_DIR, "predictions")

# Logs
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
TENSORBOARD_DIR = os.path.join(LOGS_DIR, "tensorboard")

# File names
MODEL_FILENAME = "brain_tumor_model.keras"
TRAINING_HISTORY_PLOT = "training_history.png"
CONFUSION_MATRIX_PLOT = "confusion_matrix.png"
PREDICTIONS_GRID_PLOT = "predictions_grid.png"
CONFIDENCE_DIST_PLOT = "confidence_distribution.png"

def create_output_dirs():
    """Create all necessary output directories."""
    dirs = [
        OUTPUT_DIR,
        MODELS_DIR,
        CHECKPOINTS_DIR,
        VISUALIZATIONS_DIR,
        TRAINING_PLOTS_DIR,
        EVALUATION_PLOTS_DIR,
        PREDICTION_PLOTS_DIR,
        LOGS_DIR,
        TENSORBOARD_DIR
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def get_model_path(model_name=MODEL_FILENAME):
    """Get the full path for saving/loading a model."""
    return os.path.join(MODELS_DIR, model_name)

def get_checkpoint_path(epoch):
    """Get the path for a model checkpoint at a specific epoch."""
    return os.path.join(CHECKPOINTS_DIR, f"checkpoint_epoch_{epoch}.keras")

def get_visualization_path(plot_name, plot_type="training"):
    """
    Get the path for a visualization file.
    
    Args:
        plot_name: Name of the plot file
        plot_type: Type of plot (training, evaluation, or predictions)
    
    Returns:
        Full path for the visualization file
    """
    plot_dirs = {
        "training": TRAINING_PLOTS_DIR,
        "evaluation": EVALUATION_PLOTS_DIR,
        "predictions": PREDICTION_PLOTS_DIR
    }
    return os.path.join(plot_dirs[plot_type], plot_name) 