"""
Model Interpretation Utilities

This module provides tools for interpreting the model's predictions
using SHAP (SHapley Additive exPlanations) values.
"""

import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os

from configs.model_config import CLASS_NAMES
from configs.output_config import get_visualization_path

class ModelInterpreter:
    """Class for interpreting model predictions using SHAP."""
    
    def __init__(self, model: tf.keras.Model):
        """
        Initialize the model interpreter.
        
        Args:
            model: Trained TensorFlow model to interpret
        """
        self.model = model
        self.explainer = None
    
    def _prepare_background_data(
        self,
        background_data: np.ndarray,
        max_samples: int = 100
    ) -> np.ndarray:
        """
        Prepare background data for SHAP analysis.
        
        Args:
            background_data: Dataset to use as background
            max_samples: Maximum number of samples to use
            
        Returns:
            Prepared background data
        """
        if len(background_data) > max_samples:
            indices = np.random.choice(
                len(background_data),
                max_samples,
                replace=False
            )
            background_data = background_data[indices]
        return background_data
    
    def initialize_explainer(
        self,
        background_data: np.ndarray,
        max_background_samples: int = 100
    ) -> None:
        """
        Initialize the SHAP explainer.
        
        Args:
            background_data: Dataset to use as background
            max_background_samples: Maximum number of background samples
        """
        background = self._prepare_background_data(
            background_data,
            max_background_samples
        )
        self.explainer = shap.DeepExplainer(self.model, background)
    
    def explain_image(
        self,
        image: np.ndarray,
        class_index: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate SHAP values for a single image.
        
        Args:
            image: Input image to explain
            class_index: Specific class to explain (None for all classes)
            
        Returns:
            Tuple of (shap_values, class_names)
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
            
        shap_values = self.explainer.shap_values(image)
        
        if class_index is not None:
            return shap_values[class_index], [CLASS_NAMES[class_index]]
        return shap_values, CLASS_NAMES
    
    def plot_shap_summary(
        self,
        test_images: np.ndarray,
        max_display: int = 10,
        plot_type: str = "dot",
        class_index: Optional[int] = None,
        filename: str = "shap_summary.png"
    ) -> None:
        """
        Create and save a SHAP summary plot.
        
        Args:
            test_images: Set of test images to explain
            max_display: Maximum number of features to display
            plot_type: Type of summary plot ('dot' or 'bar')
            class_index: Specific class to explain (None for all classes)
            filename: Name of the output file
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")
            
        # Prepare test images
        test_sample = self._prepare_background_data(test_images, max_display)
        
        # Generate SHAP values
        shap_values = self.explainer.shap_values(test_sample)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        if class_index is not None:
            shap.summary_plot(
                shap_values[class_index],
                test_sample,
                plot_type=plot_type,
                class_names=[CLASS_NAMES[class_index]],
                show=False
            )
        else:
            shap.summary_plot(
                shap_values,
                test_sample,
                plot_type=plot_type,
                class_names=CLASS_NAMES,
                show=False
            )
        
        # Save plot
        plt.savefig(
            get_visualization_path(filename, "evaluation"),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
    
    def plot_shap_image_importance(
        self,
        image: np.ndarray,
        class_index: int,
        filename: str = "shap_importance.png"
    ) -> None:
        """
        Create and save a SHAP image importance plot for a specific class.
        
        Args:
            image: Input image to explain
            class_index: Class index to explain
            filename: Name of the output file
        """
        shap_values, _ = self.explain_image(image, class_index)
        
        plt.figure(figsize=(10, 5))
        
        # Original image
        plt.subplot(121)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        # SHAP values
        plt.subplot(122)
        shap_abs = np.abs(shap_values).mean(axis=-1)
        plt.imshow(shap_abs, cmap='hot')
        plt.title(f"Feature Importance for {CLASS_NAMES[class_index]}")
        plt.colorbar(label='|SHAP value|')
        plt.axis('off')
        
        plt.savefig(
            get_visualization_path(filename, "evaluation"),
            bbox_inches='tight',
            dpi=300
        )
        plt.close() 