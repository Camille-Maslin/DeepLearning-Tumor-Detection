"""
Unit tests for the visualization utilities.

This module contains test cases for the ModelVisualizer class and its methods.
"""

import unittest
import os
import numpy as np
import shutil
from src.utils.visualization import ModelVisualizer

class TestModelVisualizer(unittest.TestCase):
    """Test cases for ModelVisualizer class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = "test_visualizations"
        self.visualizer = ModelVisualizer(self.test_dir)
        
        # Create mock data
        self.mock_history = {
            'accuracy': [0.7, 0.8, 0.9],
            'val_accuracy': [0.6, 0.7, 0.8],
            'loss': [0.5, 0.4, 0.3],
            'val_loss': [0.6, 0.5, 0.4]
        }
        
        self.mock_images = np.random.rand(9, 256, 256, 3) * 255
        self.mock_labels = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0])
        self.mock_predictions = [
            ('glioma', 95.5),
            ('meningioma', 87.3),
            ('notumor', 92.1),
            ('pituitary', 88.9),
            ('glioma', 91.2),
            ('meningioma', 89.4),
            ('notumor', 93.7),
            ('pituitary', 90.1),
            ('glioma', 94.3)
        ]

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_directory_creation(self):
        """Test if visualization directory is created correctly."""
        self.assertTrue(os.path.exists(self.test_dir))

    def test_plot_training_history(self):
        """Test training history plotting."""
        filename = 'test_history.png'
        self.visualizer.plot_training_history(self.mock_history, filename)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, filename)))

    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        filename = 'test_confusion.png'
        y_true = [0, 1, 2, 3, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 3, 1, 1, 2, 2]
        
        self.visualizer.plot_confusion_matrix(y_true, y_pred, filename)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, filename)))

    def test_plot_sample_predictions(self):
        """Test sample predictions plotting."""
        filename = 'test_predictions.png'
        
        self.visualizer.plot_sample_predictions(
            self.mock_images,
            self.mock_labels,
            self.mock_predictions,
            filename
        )
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, filename)))

    def test_plot_correct_predictions_confidence(self):
        """Test prediction confidence plotting."""
        filename = 'test_confidence.png'
        confidences = [95.5, 87.3, 92.1, 88.9, 91.2]
        is_correct = [True, False, True, True, False]
        
        self.visualizer.plot_correct_predictions_confidence(
            confidences,
            is_correct,
            filename
        )
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, filename)))

if __name__ == '__main__':
    unittest.main() 