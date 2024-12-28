"""
Brain Tumor Image Data Augmentation Module

This module provides functionality for augmenting brain tumor images using various
techniques to enhance the dataset for deep learning model training.

Features:
- Configurable image size and augmentation parameters
- Multiple augmentation techniques (flip, rotation, zoom, contrast)
- Support for various image formats
- Automatic handling of different color spaces (RGB, RGBA, grayscale)

Author: Camille Maslin
Date: December 2023
"""

import tensorflow as tf
from tensorflow.keras import layers
import os
from PIL import Image
import numpy as np
from typing import Tuple, Optional

class ImageAugmentor:
    """A class to handle image augmentation operations for medical imaging."""
    
    def __init__(
        self,
        image_size: int = 256,
        augmentation_factor: int = 2,
        rotation_factor: float = 0.1,
        zoom_factor: float = 0.1,
        contrast_factor: float = 0.2
    ):
        """
        Initialize the ImageAugmentor.

        Args:
            image_size (int): Target size for images (both width and height)
            augmentation_factor (int): Number of augmented images to generate per original
            rotation_factor (float): Maximum rotation angle as a fraction of 2π
            zoom_factor (float): Maximum zoom factor
            contrast_factor (float): Maximum contrast adjustment factor
        """
        self.image_size = image_size
        self.augmentation_factor = augmentation_factor
        
        # Define augmentation pipeline
        self.augmentation_pipeline = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(rotation_factor, fill_mode='constant', interpolation='bilinear'),
            layers.RandomZoom(zoom_factor, fill_mode='constant'),
            layers.RandomContrast(contrast_factor)
        ])

    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        """
        Preprocess an image for augmentation.

        Args:
            img (PIL.Image): Input image

        Returns:
            np.ndarray: Preprocessed image array
        """
        img = img.resize((self.image_size, self.image_size))
        
        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img_array = np.array(img)

        # Handle different color spaces
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[2] == 1:
            img_array = np.repeat(img_array, 3, axis=2)
        elif img_array.shape[2] == 4:
            img_array = img_array[:,:,:3]

        return img_array

    def augment_image(self, img_array: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to a single image.

        Args:
            img_array (np.ndarray): Input image array

        Returns:
            np.ndarray: Augmented image array
        """
        augmented_img = self.augmentation_pipeline(
            tf.expand_dims(img_array, 0),
            training=True
        )
        augmented_img = tf.squeeze(augmented_img).numpy().astype("uint8")
        
        # Remove artifacts
        augmented_img = np.where(augmented_img < 5, 0, augmented_img)
        
        return augmented_img

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        file_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')
    ) -> None:
        """
        Process all images in a directory and its subdirectories.

        Args:
            input_dir (str): Path to input directory containing original images
            output_dir (str): Path to output directory for augmented images
            file_extensions (tuple): Tuple of valid file extensions to process
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for subdir, _, files in os.walk(input_dir):
            relative_path = os.path.relpath(subdir, input_dir)
            current_output_dir = os.path.join(output_dir, relative_path)
            
            if not os.path.exists(current_output_dir):
                os.makedirs(current_output_dir)

            for file in files:
                if file.lower().endswith(file_extensions):
                    self._process_single_image(
                        os.path.join(subdir, file),
                        current_output_dir,
                        file
                    )

    def _process_single_image(
        self,
        image_path: str,
        output_dir: str,
        filename: str
    ) -> None:
        """
        Process a single image file.

        Args:
            image_path (str): Path to the input image
            output_dir (str): Directory to save the augmented images
            filename (str): Original filename
        """
        try:
            img = Image.open(image_path)
            img_array = self._preprocess_image(img)

            # Save original image
            Image.fromarray(img_array).save(os.path.join(output_dir, filename))

            # Generate and save augmented versions
            for i in range(self.augmentation_factor):
                augmented_img = self.augment_image(img_array)
                output_path = os.path.join(
                    output_dir,
                    f"aug_{i}_{filename}"
                )
                Image.fromarray(augmented_img).save(output_path)

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    augmentor = ImageAugmentor(
        image_size=256,
        augmentation_factor=2
    )
    
    # Define your input and output directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    input_dir = os.path.join(project_root, "data", "raw")
    output_dir = os.path.join(project_root, "data", "augmented")
    
    augmentor.process_directory(input_dir, output_dir) 