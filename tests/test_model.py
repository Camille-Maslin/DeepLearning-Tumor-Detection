"""
Unit tests for the Brain Tumor CNN model.

This module contains test cases for the BrainTumorCNN class and its methods.
"""

import unittest
import tensorflow as tf
import numpy as np
import os
from src.models.brain_tumor_cnn.model import BrainTumorCNN
from configs.model_config import IMAGE_SIZE, CHANNELS, N_CLASSES

class TestBrainTumorCNN(unittest.TestCase):
    """Test cases for BrainTumorCNN class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model = BrainTumorCNN()
        self.test_image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        self.test_batch = np.random.rand(4, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

    def test_model_initialization(self):
        """Test if model is correctly initialized."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.model)

    def test_model_architecture(self):
        """Test if model architecture is correct."""
        # Test input shape
        self.assertEqual(
            self.model.model.input_shape,
            (None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        )
        # Test output shape
        self.assertEqual(
            self.model.model.output_shape,
            (None, N_CLASSES)
        )

    def test_predict_single_image(self):
        """Test single image prediction."""
        class_name, confidence = self.model.predict_single_image(self.test_image)
        
        # Test return types
        self.assertIsInstance(class_name, str)
        self.assertIsInstance(confidence, float)
        
        # Test confidence range
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 100)

    def test_predict_batch(self):
        """Test batch prediction."""
        pred_classes, confidences = self.model.predict_batch(self.test_batch)
        
        # Test output shapes
        self.assertEqual(len(pred_classes), 4)
        self.assertEqual(len(confidences), 4)
        
        # Test output types and ranges
        self.assertTrue(np.issubdtype(pred_classes.dtype, np.integer))
        self.assertTrue(np.all(pred_classes >= 0))
        self.assertTrue(np.all(pred_classes < N_CLASSES))
        self.assertTrue(np.all(confidences >= 0))
        self.assertTrue(np.all(confidences <= 1))

    def test_save_and_load(self):
        """Test model saving and loading."""
        # Create temporary file path
        temp_path = "temp_model.keras"
        
        try:
            # Save model
            self.model.save(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load model
            loaded_model = BrainTumorCNN.load(temp_path)
            self.assertIsNotNone(loaded_model)
            
            # Test predictions consistency
            original_pred = self.model.predict_single_image(self.test_image)
            loaded_pred = loaded_model.predict_single_image(self.test_image)
            self.assertEqual(original_pred[0], loaded_pred[0])
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    unittest.main() 