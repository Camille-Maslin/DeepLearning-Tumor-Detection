"""
Visualization Utilities

This module provides functions for visualizing model training results,
predictions, and evaluation metrics.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Tuple
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from configs.model_config import CLASS_NAMES

class ModelVisualizer:
    """Class for visualizing model results and predictions."""
    
    def __init__(self, save_dir: str):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save visualization results
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        filename: str = 'training_results.png'
    ) -> None:
        """
        Plot training and validation metrics.
        
        Args:
            history: Training history dictionary
            filename: Name of the output file
        """
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
    
    def plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        filename: str = 'confusion_matrix.png'
    ) -> None:
        """
        Create and plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            filename: Name of the output file
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
    
    def plot_sample_predictions(
        self,
        images: np.ndarray,
        true_labels: np.ndarray,
        predictions: List[Tuple[str, float]],
        filename: str = 'predictions.png'
    ) -> None:
        """
        Plot a grid of sample predictions.
        
        Args:
            images: Batch of images
            true_labels: True class indices
            predictions: List of (predicted_class, confidence) tuples
            filename: Name of the output file
        """
        plt.figure(figsize=(15, 15))
        for i in range(min(9, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].astype("uint8"))
            
            predicted_class, confidence = predictions[i]
            actual_class = CLASS_NAMES[true_labels[i]]
            
            plt.title(
                f"Actual: {actual_class}\n"
                f"Predicted: {predicted_class}\n"
                f"Confidence: {confidence}%"
            )
            plt.axis("off")
        
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
    
    def plot_correct_predictions_confidence(
        self,
        confidences: List[float],
        is_correct: List[bool],
        filename: str = 'prediction_confidence.png'
    ) -> None:
        """
        Plot histogram of prediction confidences for correct and incorrect predictions.
        
        Args:
            confidences: List of prediction confidences
            is_correct: List of boolean values indicating if predictions were correct
            filename: Name of the output file
        """
        correct_conf = [conf for conf, correct in zip(confidences, is_correct) if correct]
        incorrect_conf = [conf for conf, correct in zip(confidences, is_correct) if not correct]
        
        plt.figure(figsize=(10, 6))
        plt.hist(correct_conf, alpha=0.5, label='Correct Predictions', bins=20)
        plt.hist(incorrect_conf, alpha=0.5, label='Incorrect Predictions', bins=20)
        plt.xlabel('Confidence (%)')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Confidences')
        plt.legend()
        
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close() 