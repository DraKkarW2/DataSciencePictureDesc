import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt

data_dir = r"C:\Users\steve\Documents\GitHub\DataSciencePictureDesc\DataSets\Rebanced_DataSets\augmentation"

# Paramètres de base
input_shape = (128, 128, 3)
batch_size = 32
epochs = 30

# Prétraitement des données
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_set = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    seed=42
)

validation_set = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    seed=42
)

# Fonction de construction du modèle
def build_model(hp):
    encoder_inputs = Input(shape=input_shape)

    x = Conv2D(hp.Choice('conv_1_filter', [32, 64, 128]), 
               (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.03))(encoder_inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(hp.Choice('dropout_1', [0.4, 0.5, 0.6]))(x)

    x = Conv2D(hp.Choice('conv_2_filter', [64, 128, 256]), 
               (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.03))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(hp.Choice('dropout_2', [0.4, 0.5, 0.6]))(x)

    x = Conv2D(hp.Choice('conv_3_filter', [128, 256, 512]), 
               (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.03))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(hp.Choice('dropout_3', [0.4, 0.5, 0.6]))(x)

    x = Flatten()(x)
    latent_space = Dense(hp.Choice('dense_units', [128, 256, 512]), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.03))(x)

    x = Dense(hp.Choice('dense_1_units', [64, 128, 256]), activation='relu')(latent_space)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(encoder_inputs, outputs)

    model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-3, 1e-4, 5e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Recherche des hyperparamètres optimaux avec Grid Search
tuner = kt.GridSearch(
    build_model,
    objective='val_loss',  # Optimiser la validation loss
    max_trials=5,  
    executions_per_trial=1,
    directory='my_dir',
    project_name='grid_search_autoencoder'
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
    CSVLogger('training_log.csv', append=True)  # Enregistre les logs
]

# Appel du tuner pour trouver les meilleurs hyperparamètres
tuner.search(train_set, validation_data=validation_set, epochs=10, callbacks=callbacks)

# Meilleurs hyperparamètres
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Meilleurs hyperparamètres : {best_hps.values}")

# Entraînement du modèle avec les meilleurs hyperparamètres
best_model = tuner.get_best_models(num_models=1)[0]
history = best_model.fit(train_set, epochs=epochs, validation_data=validation_set, callbacks=callbacks)

# Si tu veux récupérer la validation accuracy à tout moment
val_accuracy = best_model.evaluate(validation_set)
print(f"Validation Accuracy: {val_accuracy[1] * 100:.2f}%")

# Si tu veux afficher les courbes d'entraînement à tout moment
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Affichage des courbes
plot_training_history(history)
