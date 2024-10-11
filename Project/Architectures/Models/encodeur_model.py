import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import os

# Paramètres de base
input_shape = (128, 128, 3)
batch_size = 32
epochs = 10
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = r"C:\Users\steve\Documents\GitHub\DataSciencePictureDesc\DataSets\Rebanced_DataSets\Random_DataSets"

# Préparation des données avec rescaling et augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary',  # Classification binaire
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary',  # Classification binaire
    subset='validation'
)

# Construction du modèle d'encodeur avec classification
def build_encoder_classifier(input_shape):
    model = Sequential()

    # Partie Encodeur
    model.add(Rescaling(1./255, input_shape=input_shape))  # Normalisation
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Partie Goulot d'étranglement (extraire les caractéristiques)
    model.add(Flatten())

    # Partie Classification avec régularisation et réduction de la complexité
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))  # Réduction de la taille et ajout de L2
    model.add(Dropout(0.3))  
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))  # Réduction à 32 unités

    model.add(Dense(1, activation='sigmoid'))  # Couche de sortie binaire pour la classification (0 ou 1)

    return model

# Créer le modèle d'encodeur + classification
model = build_encoder_classifier(input_shape)

# Compilation du modèle avec un learning rate réduit
optimizer = Adam(learning_rate=0.0001)  # Réduire le taux d'apprentissage
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Affichage du résumé du modèle
model.summary()

# Entraîner le modèle
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Sauvegarder les poids du modèle après l'entraînement
model.save_weights('classifier_clean_random.weights.h5')

# Fonction pour afficher les courbes d'entraînement et de validation
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Afficher les courbes
plot_training_history(history)
