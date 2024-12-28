"""
Unit tests for model interpretation utilities.

This module contains test cases for the ModelInterpreter class.
"""

import unittest
import numpy as np
import tensorflow as tf
import os
import shutil
from src.utils.model_interpretation import ModelInterpreter
from configs.model_config import IMAGE_SIZE, CHANNELS, N_CLASSES

class TestModelInterpreter(unittest.TestCase):
    """Test cases for ModelInterpreter class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple model for testing
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
            tf.keras.layers.Conv2D(16, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(N_CLASSES, activation='softmax')
        ])
        
        self.interpreter = ModelInterpreter(self.model)
        
        # Create test data
        self.test_image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        self.test_batch = np.random.rand(4, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        
        # Create output directory
        self.test_output_dir = "test_interpretation_output"
        os.makedirs(self.test_output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
    
    def test_initialization(self):
        """Test if interpreter is correctly initialized."""
        self.assertIsNotNone(self.interpreter)
        self.assertIsNotNone(self.interpreter.model)
        self.assertIsNone(self.interpreter.explainer)
    
    def test_prepare_background_data(self):
        """Test background data preparation."""
        # Test with small dataset
        small_data = np.random.rand(50, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        prepared_small = self.interpreter._prepare_background_data(small_data, 100)
        self.assertEqual(len(prepared_small), 50)
        
        # Test with large dataset
        large_data = np.random.rand(200, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        prepared_large = self.interpreter._prepare_background_data(large_data, 100)
        self.assertEqual(len(prepared_large), 100)
    
    def test_initialize_explainer(self):
        """Test explainer initialization."""
        background_data = np.random.rand(10, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        self.interpreter.initialize_explainer(background_data)
        self.assertIsNotNone(self.interpreter.explainer)
    
    def test_explain_image_without_initialization(self):
        """Test explaining image without initializing explainer."""
        with self.assertRaises(ValueError):
            self.interpreter.explain_image(self.test_image)
    
    def test_explain_image(self):
        """Test image explanation generation."""
        # Initialize explainer
        background_data = np.random.rand(10, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        self.interpreter.initialize_explainer(background_data)
        
        # Test single class explanation
        shap_values, class_names = self.interpreter.explain_image(
            self.test_image,
            class_index=0
        )
        self.assertEqual(len(class_names), 1)
        self.assertEqual(shap_values.shape[1:], (IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        
        # Test all classes explanation
        shap_values, class_names = self.interpreter.explain_image(self.test_image)
        self.assertEqual(len(class_names), N_CLASSES)
        self.assertEqual(len(shap_values), N_CLASSES)
    
    def test_plot_generation(self):
        """Test if plots are generated correctly."""
        # Initialize explainer
        background_data = np.random.rand(10, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        self.interpreter.initialize_explainer(background_data)
        
        # Test summary plot
        test_images = np.random.rand(5, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        self.interpreter.plot_shap_summary(
            test_images,
            max_display=5,
            filename=os.path.join(self.test_output_dir, "test_summary.png")
        )
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "test_summary.png")))
        
        # Test importance plot
        self.interpreter.plot_shap_image_importance(
            self.test_image,
            class_index=0,
            filename=os.path.join(self.test_output_dir, "test_importance.png")
        )
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "test_importance.png")))

if __name__ == '__main__':
    unittest.main() 