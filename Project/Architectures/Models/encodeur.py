import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration des paramètres
input_shape = (128, 128, 3)
batch_size = 16
epochs = 10
dataset_dir = r"C:\Users\steve\Documents\GitHub\DataSciencePictureDesc\DataSets\Rebanced_DataSets\augmentation"

# Génération des données avec normalisation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Générateur d'entraînement pour l'autoencodeur
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode=None,  # Pas d'étiquettes pour l'autoencodeur
    subset='training',
    shuffle=False  # Désactivation du mélange des données pour simplifier
)

# Générateur de validation pour l'autoencodeur
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode=None,  # Pas d'étiquettes pour l'autoencodeur
    subset='validation',
    shuffle=False
)

# Vérification des batchs générés par le générateur
for i, x_batch in enumerate(train_generator):
    print(f"Batch {i+1} - Shape of training batch: {x_batch.shape}")
    print(f"First batch sample: {x_batch[0]}")  # Afficher un exemple d'image
    if i >= 4:  # Tester sur les 5 premiers batchs
        break

# Architecture de l'autoencodeur
input_img = Input(shape=input_shape)

# Encodeur
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Décodeur
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Création de l'autoencodeur complet
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Afficher le résumé du modèle
autoencoder.summary()

# Test avec des données synthétiques pour isoler le problème
synthetic_data = np.random.rand(100, 128, 128, 3)  # 100 images aléatoires
synthetic_labels = synthetic_data  # Pour l'autoencodeur

# Entraîner l'autoencodeur avec les données synthétiques
history_autoencoder = autoencoder.fit(
    synthetic_data, synthetic_labels,
    epochs=epochs,
    batch_size=batch_size
)

# Fonction pour afficher les courbes d'entraînement de l'autoencodeur
def plot_autoencoder_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

# Afficher les courbes de l'autoencodeur
plot_autoencoder_history(history_autoencoder)
