"""
Brain Tumor CNN Model

This module implements a CNN model for brain tumor classification using TensorFlow/Keras.
It provides a clean, object-oriented interface for model creation, training, and evaluation.
"""

import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from typing import Tuple, Optional, Dict, Any

from configs.model_config import (
    IMAGE_SIZE,
    CHANNELS,
    N_CLASSES,
    CLASS_NAMES
)

class BrainTumorCNN:
    """A CNN model for brain tumor classification."""
    
    def __init__(self):
        """Initialize the model architecture."""
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        Build and compile the CNN model.
        
        Returns:
            tf.keras.Model: Compiled model ready for training.
        """
        model = models.Sequential([
            layers.Rescaling(1./255, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(N_CLASSES, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        return model
    
    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int,
        callbacks: Optional[list] = None
    ) -> tf.keras.callbacks.History:
        """
        Train the model on the provided dataset.
        
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            epochs: Number of training epochs
            callbacks: Optional list of callbacks
            
        Returns:
            Training history
        """
        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    
    def evaluate(
        self,
        test_ds: tf.data.Dataset
    ) -> Tuple[float, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_ds: Test dataset
            
        Returns:
            Tuple of (loss, accuracy)
        """
        return self.model.evaluate(test_ds)
    
    def predict_single_image(
        self,
        img: np.ndarray
    ) -> Tuple[str, float]:
        """
        Make a prediction for a single image.
        
        Args:
            img: Input image array
            
        Returns:
            Tuple of (predicted_class_name, confidence)
        """
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        return CLASS_NAMES[predicted_class_idx], round(confidence * 100, 2)
    
    def predict_batch(
        self,
        batch_images: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions for a batch of images.
        
        Args:
            batch_images: Batch of input images
            
        Returns:
            Tuple of (predicted_class_indices, confidences)
        """
        predictions = self.model.predict(batch_images)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        return predicted_classes, confidences
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'BrainTumorCNN':
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            BrainTumorCNN instance with loaded weights
        """
        instance = cls()
        instance.model = tf.keras.models.load_model(filepath)
        return instance
    
    def get_summary(self) -> str:
        """
        Get a string summary of the model architecture.
        
        Returns:
            Model summary as string
        """
        return self.model.summary() 