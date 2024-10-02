import tensorflow as tf
from tensorflow.keras import layers
import os
from PIL import Image
import numpy as np

# Configuration
IMAGE_SIZE = 256
AUGMENTATION_FACTOR = 2

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, "data", "base_data")
output_dir = os.path.join(os.path.dirname(data_dir), "data_augmente_data")

# Data Augmentation Pipeline with controlled border handling
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1, fill_mode='constant', interpolation='bilinear'),  # Reduced rotation, constant fill
    layers.RandomZoom(0.1, fill_mode='constant'),  # Reduced zoom, constant fill
    layers.RandomContrast(0.2)
])

# Fonction pour créer un dossier si nécessaire
def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Créer un nouveau dataset avec des images augmentées
def augment_and_save_images(dataset_dir, output_dir, augmentation_factor):
    create_dir(output_dir)

    for subdir, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):  # Assurez-vous que ce sont bien des fichiers images
                image_path = os.path.join(subdir, file)
                img = Image.open(image_path)
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                
                # Convert RGBA to RGB if necessary
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                img_array = np.array(img)

                # Vérifier si l'image est en niveaux de gris et la convertir en RGB si nécessaire
                if len(img_array.shape) == 2:
                    img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[2] == 1:
                    img_array = np.repeat(img_array, 3, axis=2)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:,:,:3]  # Keep only RGB channels

                label_dir = os.path.join(output_dir, os.path.basename(subdir))
                create_dir(label_dir)

                # Sauvegarder l'image originale
                Image.fromarray(img_array).save(os.path.join(label_dir, file))

                # Appliquer les augmentations et sauvegarder les images modifiées
                for i in range(augmentation_factor):
                    augmented_img = data_augmentation(tf.expand_dims(img_array, 0), training=True)
                    augmented_img = tf.squeeze(augmented_img).numpy().astype("uint8")
                    
                    # Post-processing to remove potential artifacts
                    augmented_img = np.where(augmented_img < 5, 0, augmented_img)  # Remove very dark artifacts
                    
                    augmented_img_pil = Image.fromarray(augmented_img)

                    # Sauvegarder l'image augmentée avec un nouveau nom
                    augmented_img_pil.save(os.path.join(label_dir, f"aug_{i}_{file}"))

# Appeler la fonction pour générer des images augmentées
augment_and_save_images(data_dir, output_dir, AUGMENTATION_FACTOR)

print(f"Les images augmentées ont été sauvegardées dans : {output_dir}")
