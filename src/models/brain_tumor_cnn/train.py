"""
Training Script for Brain Tumor CNN Model

This script handles the training pipeline for the brain tumor classification model,
including data loading, model training, evaluation, and results visualization.
"""

import os
import tensorflow as tf
import numpy as np
from typing import Tuple

from configs.model_config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    EPOCHS,
    TRAIN_SPLIT,
    VAL_SPLIT,
    SHUFFLE_BUFFER_SIZE,
    DATA_DIR,
    TEST_DATA_DIR,
    MODEL_SAVE_PATH,
    RESULTS_DIR
)
from src.models.brain_tumor_cnn.model import BrainTumorCNN
from src.utils.visualization import ModelVisualizer
from src.utils.model_interpretation import ModelInterpreter

def load_and_prepare_data(
    data_dir: str,
    batch_size: int,
    image_size: int
) -> tf.data.Dataset:
    """
    Load and prepare the dataset for training.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for dataset
        image_size: Target image size
        
    Returns:
        Prepared dataset
    """
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        shuffle=True,
        image_size=(image_size, image_size),
        batch_size=batch_size
    )

def split_dataset(
    dataset: tf.data.Dataset,
    train_split: float = 0.9,
    val_split: float = 0.1
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: Input dataset
        train_split: Proportion for training
        val_split: Proportion for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    ds_size = len(dataset)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    
    return train_ds, val_ds

def optimize_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Optimize dataset performance.
    
    Args:
        dataset: Input dataset
        
    Returns:
        Optimized dataset
    """
    AUTOTUNE = tf.data.AUTOTUNE
    return dataset.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)

def generate_model_interpretations(
    model: BrainTumorCNN,
    test_dataset: tf.data.Dataset,
    max_samples: int = 100
) -> None:
    """
    Generate SHAP interpretations for the model.
    
    Args:
        model: Trained model to interpret
        test_dataset: Test dataset for interpretations
        max_samples: Maximum number of samples to use
    """
    print("Generating model interpretations...")
    
    # Collect background data
    background_images = []
    for images, _ in test_dataset.take(5):  # Take 5 batches for background
        background_images.extend(images.numpy())
    background_images = np.array(background_images)
    
    # Initialize interpreter
    interpreter = ModelInterpreter(model.model)
    interpreter.initialize_explainer(background_images)
    
    # Generate interpretations for each class
    for class_idx, class_name in enumerate(CLASS_NAMES):
        print(f"Generating interpretations for class: {class_name}")
        
        # Get some test images
        test_images = []
        test_labels = []
        for images, labels in test_dataset:
            for img, label in zip(images.numpy(), labels.numpy()):
                if label == class_idx:
                    test_images.append(img)
                    test_labels.append(label)
                if len(test_images) >= max_samples:
                    break
            if len(test_images) >= max_samples:
                break
        
        if test_images:
            test_images = np.array(test_images)
            
            # Generate SHAP summary plot for this class
            interpreter.plot_shap_summary(
                test_images,
                class_index=class_idx,
                filename=f"shap_summary_{class_name}.png"
            )
            
            # Generate importance plot for first image of this class
            interpreter.plot_shap_image_importance(
                test_images[0],
                class_idx,
                filename=f"shap_importance_{class_name}.png"
            )

def main():
    """Main training function."""
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize visualizer
    visualizer = ModelVisualizer(RESULTS_DIR)
    
    # Load and prepare datasets
    print("Loading datasets...")
    dataset = load_and_prepare_data(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
    test_dataset = load_and_prepare_data(TEST_DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
    
    # Split and optimize datasets
    train_ds, val_ds = split_dataset(dataset, TRAIN_SPLIT, VAL_SPLIT)
    train_ds = optimize_dataset(train_ds)
    val_ds = optimize_dataset(val_ds)
    test_dataset = optimize_dataset(test_dataset)
    
    # Create and train model
    print("Initializing model...")
    model = BrainTumorCNN()
    print(model.get_summary())
    
    print("Starting training...")
    history = model.train(train_ds, val_ds, EPOCHS)
    
    # Save model
    print("Saving model...")
    model.save(MODEL_SAVE_PATH)
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_accuracy:.2f}")
    
    # Visualize results
    print("Generating visualizations...")
    visualizer.plot_training_history(history.history)
    
    # Generate predictions for confusion matrix
    y_true = []
    y_pred = []
    all_confidences = []
    is_correct = []
    
    for images, labels in test_dataset:
        pred_classes, confidences = model.predict_batch(images)
        y_true.extend(labels.numpy())
        y_pred.extend(pred_classes)
        all_confidences.extend(confidences * 100)
        is_correct.extend(pred_classes == labels.numpy())
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(y_true, y_pred)
    
    # Plot confidence distribution
    visualizer.plot_correct_predictions_confidence(all_confidences, is_correct)
    
    # Plot sample predictions
    for images, labels in test_dataset.take(1):
        predictions = [
            model.predict_single_image(img.numpy())
            for img in images[:9]
        ]
        visualizer.plot_sample_predictions(
            images.numpy(),
            labels.numpy(),
            predictions
        )
    
    # After model evaluation and before visualization
    print("Generating model interpretations...")
    generate_model_interpretations(model, test_dataset)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 