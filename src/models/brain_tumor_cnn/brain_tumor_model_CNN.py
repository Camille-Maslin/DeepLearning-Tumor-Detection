"""
Brain Tumor Classification Model using CNN.

This script implements a deep learning model for classifying brain tumor images
using a custom CNN architecture. It includes data loading,
model creation, training, evaluation, and result visualization.

Dataset link : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Author: Camille Maslin
Date:
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# Environment setup
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

# Print TensorFlow and Keras versions
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Configuration
BATCH_SIZE = 32
IMAGE_SIZE = 256 
CHANNELS = 3
EPOCHS = 20

# Directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
data_dir = os.path.join(parent_dir, "data", "base_data")
test_data_dir = os.path.join(parent_dir, "data", "testing_images")

def load_data(data_dir, batch_size, image_size):
    """
    Load and preprocess the image dataset.
    
    Args:
        data_dir (str): Path to the data directory.
        batch_size (int): Batch size for dataset.
        image_size (tuple): Target size for images.
    
    Returns:
        tf.data.Dataset: Preprocessed dataset.
    """
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        shuffle=True,
        image_size=(image_size, image_size),
        batch_size=batch_size
    )

# Load datasets
dataset = load_data(data_dir, BATCH_SIZE, IMAGE_SIZE)
test_dataset = load_data(test_data_dir, BATCH_SIZE, IMAGE_SIZE)

def split_dataset(ds, train_split=0.9, val_split=0.1, shuffle=True, shuffle_size=10000):
    """
    Split the dataset into training and validation sets.
    
    Args:
        ds (tf.data.Dataset): Input dataset.
        train_split (float): Proportion of data for training.
        val_split (float): Proportion of data for validation.
        shuffle (bool): Whether to shuffle the dataset.
        shuffle_size (int): Buffer size for shuffling.
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    
    return train_ds, val_ds

# Split dataset
train_ds, val_ds = split_dataset(dataset)

# Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER_SIZE = 1000

 # Define classes names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
n_classes = len(class_names)

train_ds = train_ds.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

def create_model():
    """
    Create and return the CNN model for brain tumor classification.
    
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
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    return model

# Create and train the model
model = create_model()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy:.2f}")

def plot_training_history(history):
    """
    Plot the training and validation accuracy/loss.
    
    Args:
        history: Training history object.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(os.path.join(current_dir, 'training_results.png'))
    plt.close()

# Plot training history
plot_training_history(history)

def predict(model, img):
    """
    Make a prediction for a single image.
    
    Args:
        model (tf.keras.Model): Trained model.
        img (np.array): Input image.
    
    Returns:
        tuple: (predicted_class, confidence)
    """
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

def plot_sample_predictions(model, test_dataset):
    """
    Plot predictions for a sample of test images.
    
    Args:
        model (tf.keras.Model): Trained model.
        test_dataset (tf.data.Dataset): Test dataset.
    """
    plt.figure(figsize=(15, 15))
    for images, labels in test_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predicted_class, confidence = predict(model, images[i].numpy())
            actual_class = class_names[labels[i]] 
            plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
            plt.axis("off")

    plt.savefig(os.path.join(current_dir, 'predictions.png'))
    plt.close()

# Plot sample predictions
plot_sample_predictions(model, test_dataset)

def create_confusion_matrix(model, test_dataset):
    """
    Create and plot the confusion matrix.
    
    Args:
        model (tf.keras.Model): Trained model.
        test_dataset (tf.data.Dataset): Test dataset.
    """
    y_pred = []
    y_true = []

    for images, labels in test_dataset:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.savefig(os.path.join(current_dir, 'confusion_matrix.png'))
    plt.close()

# Create confusion matrix
create_confusion_matrix(model, test_dataset)

def create_prediction_diagrams(model, test_dataset):
    """
    Create and save prediction diagrams.
    
    Args:
        model (tf.keras.Model): Trained model.
        test_dataset (tf.data.Dataset): Test dataset.
    """
    all_predictions = []
    all_true_labels = []
    all_confidences = []

    for images, labels in test_dataset:
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        all_predictions.extend(predicted_classes)
        all_true_labels.extend(labels.numpy())
        all_confidences.extend(confidences)

    df = pd.DataFrame({
        'True Label': [class_names[i] for i in all_true_labels],
        'Predicted Label': [class_names[i] for i in all_predictions],
        'Confidence': all_confidences
    })

    df['Correct'] = df['True Label'] == df['Predicted Label']

    # Diagram for incorrect predictions
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df[~df['Correct']], x='True Label', y='Predicted Label', 
                    hue='Confidence', size='Confidence', sizes=(20, 200),
                    palette='viridis')
    plt.title('Incorrect Predictions with Confidence')
    plt.savefig(os.path.join(current_dir, 'incorrect_predictions.png'))
    plt.close()

    # Diagram for correct predictions
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[df['Correct']], x='True Label', y='Confidence')
    plt.title('Confidence Distribution for Correct Predictions')
    plt.savefig(os.path.join(current_dir, 'correct_predictions_confidence.png'))
    plt.close()

# Create prediction diagrams
create_prediction_diagrams(model, test_dataset)

# Save the model
model.save(os.path.join(current_dir, "brain_tumor_model.keras"))

print(f"Training completed. Results and model saved in: {current_dir}")