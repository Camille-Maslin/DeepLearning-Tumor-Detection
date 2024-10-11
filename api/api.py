"""
FastAPI application for testing the brain tumor classification model.

This script allows users to upload an image and receive predictions from the trained model.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import os
import uvicorn
import sys

# Environment setup
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
app = FastAPI()

# Print TensorFlow and Keras versions
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Configuration
IMAGE_SIZE = 256
MODEL_PATH = os.path.join(os.path.dirname(__file__), "brain_tumor_model.keras")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names globally
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def prepare_image(img):
    """
    Preprocess the uploaded image for prediction.
    
    Args:
        img (np.array): Input image.
    
    Returns:
        np.array: Preprocessed image.
    """
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the class of an uploaded image.
    
    Args:
        file (UploadFile): The uploaded image file.
    
    Returns:
        JSONResponse: Prediction results.
    """
    img = await file.read()
    img = tf.image.decode_image(img, channels=3)
    img = prepare_image(img)

    # Use the model to make predictions
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(100 * (np.max(predictions)), 2)

    return JSONResponse(content={"predicted_class": predicted_class, "confidence": confidence})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)