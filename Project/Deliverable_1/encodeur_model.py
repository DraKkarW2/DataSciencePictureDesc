import keras_tuner as kt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Paramètres de base
input_shape = (128, 128, 3)  # Format d'image (128x128 avec 3 canaux pour RVB)
batch_size = 32
epochs = 10

# Chemin vers ton dataset
dataset_dir = r"C:\Users\steve\Documents\GitHub\DataSciencePictureDesc\DataSets\Rebanced_DataSets\Random_DataSets"

# Fonction pour construire le modèle CNN autoencodeur avec recherche d'hyperparamètres
def build_model(hp):
    inputs = Input(shape=input_shape)

    # Partie Encodeur : Recherche des hyperparamètres pour les couches convolutionnelles
    x = Conv2D(hp.Int('conv_1_filters', min_value=32, max_value=128, step=32),
               (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(hp.Int('conv_2_filters', min_value=64, max_value=256, step=64),
               (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(hp.Int('conv_3_filters', min_value=128, max_value=512, step=128),
               (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Goulot d'étranglement
    x = Flatten()(x)
    latent_space = Dense(hp.Int('dense_latent_units', min_value=64, max_value=512, step=64), 
                         activation='relu', name="latent_vector")(x)

    # Recherche des hyperparamètres pour la classification
    x = Dense(hp.Int('dense_units', min_value=64, max_value=512, step=64), activation='relu',
              kernel_regularizer=regularizers.l2(0.01))(latent_space)
    x = Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1))(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Sortie pour la classification binaire

    model = Model(inputs, outputs, name="cnn_autoencoder_classifier")

    # Recherche du taux d'apprentissage
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Préparation des données avec ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisation des pixels entre 0 et 1
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 80% d'entraînement, 20% de validation
)

# Générateur pour l'entraînement
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),  # Redimensionner toutes les images à 128x128
    batch_size=batch_size,
    class_mode='binary',  # Classification binaire : photo ou non
    subset='training',
    shuffle=True
)

# Générateur pour la validation
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# Callback pour l'arrêt anticipé et la sauvegarde des meilleurs poids
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint_callback = ModelCheckpoint(filepath='best_cnn_autoencoder_model.keras',  # Utilisation de l'extension .keras
                                      save_best_only=True, 
                                      monitor='val_accuracy', 
                                      mode='max', 
                                      verbose=1)

# Configurer la recherche aléatoire avec KerasTuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Réduire le nombre d'essais pour accélérer la recherche
    executions_per_trial=1,  # Exécuter chaque modèle une seule fois
    directory='random_search_results',
    project_name='photo_classifier_autoencoder'
)

# Exécuter la recherche aléatoire avec les callbacks
tuner.search(train_generator, 
             epochs=epochs, 
             validation_data=validation_generator, 
             callbacks=[early_stopping_callback, checkpoint_callback])

# Résultats de la meilleure configuration
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

# Afficher les meilleurs hyperparamètres trouvés
print(f"Meilleurs hyperparamètres trouvés : {best_hyperparameters.values}")

# Évaluer le meilleur modèle sur les données de validation
val_accuracy = best_model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_accuracy[1] * 100:.2f}%")

# Sauvegarder les poids du meilleur modèle
best_model.save_weights('final_best_cnn_autoencoder_weights.h5')

# Entraîner le meilleur modèle trouvé (en cas de besoin)
history = best_model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Affichage des courbes de performance du modèle
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Graphique de la perte (loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Graphique de la précision (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Afficher les courbes d'entraînement
plot_training_history(history)

# Fonction pour prédire sur des nouvelles images avec la probabilité que ce soit une photo ou non
def predict_and_visualize(model, generator, num_images=5):
    images, labels = next(generator)
    predictions = model.predict(images)
    
    for i in range(num_images):
        plt.imshow(images[i])
        prediction = predictions[i][0]
        true_label = 'Photo' if labels[i] == 1 else 'Non-Photo'
        predicted_label = 'Photo' if prediction > 0.5 else 'Non-Photo'
        probability = prediction if prediction > 0.5 else 1 - prediction
        plt.title(f"True: {true_label} | Pred: {predicted_label} ({probability*100:.2f}%)")
        plt.axis('off')
        plt.show()

# Afficher les prédictions de classification binaire sur quelques images
predict_and_visualize(best_model, validation_generator)
