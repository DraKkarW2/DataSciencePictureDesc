import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
# Load and preprocess images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)  
    return images  


def check_image_resolutions(folder):
    resolutions = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            resolutions.append(img.shape[:2])  # (hauteur, largeur)
    return resolutions

base_dir = os.getcwd()
relative_path = r"..\..\DataSets\Dataset_delivrable_2\Dataset"
dataset_dir = os.path.normpath(os.path.join(base_dir, relative_path))

import os
import matplotlib.pyplot as plt
from collections import Counter

def check_image_resolutions(dataset_dir):
    """
    Cette fonction parcourt un répertoire contenant des images et renvoie une liste des résolutions (hauteur, largeur).
    """
    from PIL import Image

    image_resolutions = []
    
    # Parcours des fichiers dans le répertoire
    for filename in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir, filename)
        
        try:
            # Ouverture de l'image pour obtenir sa résolution
            with Image.open(file_path) as img:
                resolution = img.size  # (largeur, hauteur)
                image_resolutions.append(resolution[::-1])  # Inverser pour avoir (hauteur, largeur)
        except Exception as e:
            print(f"Erreur lors de l'ouverture du fichier {filename}: {e}")

    return image_resolutions

# Charger les résolutions d'images
image_resolutions = check_image_resolutions(dataset_dir)

# Obtenir les résolutions uniques
unique_resolutions = set(image_resolutions)
print(f"Nombre de résolutions uniques : {len(unique_resolutions)}")

# Comptage des occurrences de chaque résolution
resolution_counter = Counter([f"{res[0]}x{res[1]}" for res in image_resolutions])

# Préparer les données pour le graphique
resolutions, counts = zip(*resolution_counter.items())  # Récupère les résolutions et les fréquences

# Tracer l'histogramme
plt.figure(figsize=(10, 5))
plt.bar(resolutions, counts)  # Utilisation de bar plot pour plus de clarté
plt.xticks(rotation=90)
plt.title("Distribution des résolutions d'image")
plt.xlabel("Résolution (Hauteur x Largeur)")
plt.ylabel("Nombre d'images")
plt.tight_layout()  # Ajuste automatiquement pour éviter le chevauchement du texte
plt.show()


# Load dataset and resize images
dataset_images = load_images_from_folder(dataset_dir)
num_images = len(dataset_images)
print(f"Nombre d'images dans notre dataset : {num_images}")

def load_images_with_check(folder):
    images = []
    corrupted_files = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)  
            else:
                corrupted_files.append(filename) 
        except Exception as e:
            print(f"Erreur lors du chargement de {filename}: {e}")
            corrupted_files.append(filename) 
    return images, corrupted_files
dataset_images, corrupted_files = load_images_with_check(dataset_dir)

# Afficher le nombre d'images corrompues
print(f"Nombre d'images corrompues ou manquantes : {len(corrupted_files)}")
def display_sample_images(images, num_images=5):
    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(images))): 
        plt.subplot(1, num_images, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)) 
        plt.axis('off')  
    plt.show()

def calculate_image_statistics(images):
   
    pixel_values = np.concatenate([img.ravel() for img in images]) 
    mean = np.mean(pixel_values)
    median = np.median(pixel_values)
    std = np.std(pixel_values)
    
    print(f"Moyenne des pixels: {mean:.4f}")
    print(f"Médiane des pixels: {median:.4f}")
    print(f"Écart-type des pixels: {std:.4f}")


def plot_color_distribution(images):
    reds = np.concatenate([img[:, :, 2].ravel() for img in images]) 
    greens = np.concatenate([img[:, :, 1].ravel() for img in images])  
    blues = np.concatenate([img[:, :, 0].ravel() for img in images])  
    
    plt.figure(figsize=(10, 5))
    plt.hist(reds, bins=50, color='red', alpha=0.6, label='Rouge')
    plt.hist(greens, bins=50, color='green', alpha=0.6, label='Vert')
    plt.hist(blues, bins=50, color='blue', alpha=0.6, label='Bleu')
    plt.title("Distribution des intensités des canaux de couleur")
    plt.xlabel("Valeur de pixel")
    plt.ylabel("Nombre de pixels")
    plt.legend()
    plt.show()
    
display_sample_images(dataset_images)


calculate_image_statistics(dataset_images)

plot_color_distribution(dataset_images)


# Fonction pour redimensionner toutes les images du dataset à une taille donnée
def resize_images(images, target_size):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, target_size)
        resized_img = resized_img.astype('float16') / 255.0  
        resized_images.append(resized_img)
    return np.array(resized_images, dtype=np.float16)  # Convertir en float16


def create_augmented_dataset(original_images, additional_images, target_size):
    num_images_to_add = target_size - len(original_images)
    additional_images_sample = random.sample(additional_images, num_images_to_add)
    augmented_dataset = original_images + additional_images_sample

    return augmented_dataset


base_dir = os.getcwd()
relative_path = r"..\..\DataSets\data_set_livrable_1\Photo"
additional_dataset_dir =  os.path.normpath(os.path.join(base_dir, relative_path))

additional_images, corrupted_files = load_images_with_check(additional_dataset_dir)
print("nombre de photo dans le data set suplémentaire :",len(additional_images))
print("nombre de photo corrompu dans ce même data set :",len(corrupted_files))


augmented_dataset_1000 = create_augmented_dataset(dataset_images, additional_images, 1000)
augmented_dataset_5000 = create_augmented_dataset(dataset_images, additional_images, 5000)
augmented_dataset_10000 = create_augmented_dataset(dataset_images, additional_images, 10000)


print(len(augmented_dataset_1000))
print(len(augmented_dataset_5000))
print(len(augmented_dataset_10000))

dataset_images128 = resize_images(dataset_images, (128, 128))
dataset_images256 = resize_images(dataset_images, (256, 256))

augmented_dataset_1000_resized128 = resize_images(augmented_dataset_1000, (128, 128))
augmented_dataset_1000_resized256 = resize_images(augmented_dataset_1000, (256, 256))

augmented_dataset_5000_resized128 = resize_images(augmented_dataset_5000, (128, 128))
augmented_dataset_5000_resized256 = resize_images(augmented_dataset_5000, (256, 256))

augmented_dataset_10000_resized128 = resize_images(augmented_dataset_10000, (128, 128))
augmented_dataset_10000_resized256 = resize_images(augmented_dataset_10000, (256, 256))

print(augmented_dataset_1000_resized128.shape)
print(augmented_dataset_5000_resized256.shape)

print(dataset_images128[0][0:5, 0:5, :])

# ************************************************* Orginal data set L148 **************************************************#
# Split the original dataset resized (128x128)
train_images_orginal_R128 = dataset_images128[:int(len(dataset_images128) * 0.8)]
test_images_orginal_R128 = dataset_images128[int(len(dataset_images128) * 0.8):]
# Split the original dataset resized (256x256)
train_images_orginal_R256 = dataset_images256[:int(len(dataset_images256) * 0.8)]
test_images_orginal_R256 = dataset_images256[int(len(dataset_images256) * 0.8):]

# ************************************************* augmented data set L1000 ***********************************************#
# Split the augmented dataset 1000 resized (128x128)
train_images_augmented_L1000_R128 = augmented_dataset_1000_resized128[:int(len(augmented_dataset_1000_resized128) * 0.8)]
test_images_augmented_L1000_R128  = augmented_dataset_1000_resized128[int(len(augmented_dataset_1000_resized128) * 0.8):]

# Split the augmented dataset 1000 resized (256x256)
train_images_augmented_L1000_R256 = augmented_dataset_1000_resized256[:int(len(augmented_dataset_1000_resized256) * 0.8)]
test_images_augmented_L1000_R256  = augmented_dataset_1000_resized256[int(len(augmented_dataset_1000_resized256) * 0.8):]

# ************************************************* augmented data set L5000 ***********************************************#

# Split the augmented dataset 5000 resized (128x128)
train_images_augmented_L5000_R128 = augmented_dataset_5000_resized128[:int(len(augmented_dataset_5000_resized128) * 0.8)]
test_images_augmented_L5000_R128  = augmented_dataset_5000_resized128[int(len(augmented_dataset_5000_resized128) * 0.8):]
# Split the augmented dataset 5000 resized (256x256)
train_images_augmented_L5000_R256 = augmented_dataset_5000_resized256[:int(len(augmented_dataset_5000_resized256) * 0.8)]
test_images_augmented_L5000_R256  = augmented_dataset_5000_resized256[int(len(augmented_dataset_5000_resized256) * 0.8):]

# ************************************************* augmented data set L10 000 ***********************************************#
# Split the augmented dataset 10000 resized (128x128)
train_images_augmented_L10000_R128 = augmented_dataset_10000_resized128[:int(len(augmented_dataset_10000_resized128) * 0.8)]
test_images_augmented_L10000_R128  = augmented_dataset_10000_resized128[int(len(augmented_dataset_10000_resized128) * 0.8):]

# Split the augmented dataset 10000 resized (256x256)
train_images_augmented_L10000_R256 = augmented_dataset_10000_resized256[:int(len(augmented_dataset_10000_resized256) * 0.8)]
test_images_augmented_L10000_R256  = augmented_dataset_10000_resized256[int(len(augmented_dataset_10000_resized256) * 0.8):]


# Add noise to the images
def create_data_set_noise_batch(noise_factor, train_images, test_images, batch_size=500):
    train_noisy = []
    test_noisy = []
    
    for i in range(0, len(train_images), batch_size):
        batch_train = train_images[i:i+batch_size]
        noise_train = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=batch_train.shape).astype('float32')
        noisy_batch_train = np.clip(batch_train + noise_train, 0., 1.)
        train_noisy.append(noisy_batch_train)
    
    for i in range(0, len(test_images), batch_size):
        batch_test = test_images[i:i+batch_size]
        noise_test = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=batch_test.shape).astype('float32')
        noisy_batch_test = np.clip(batch_test + noise_test, 0., 1.)
        test_noisy.append(noisy_batch_test)
    
    return np.vstack(train_noisy), np.vstack(test_noisy)

# ************************************************* Orginal data set L148 ***********************************************#
#Add noise to the original dataset resized (128x128)
train_noisy_orginal_R128, test_noisy_orginal_R128 = create_data_set_noise_batch(noise_factor=0.5,train_images = train_images_orginal_R128,test_images = test_images_orginal_R128)
#Add noise to the original dataset resized (256x256)
train_noisy_orginal_R256, test_noisy_orginal_R256 = create_data_set_noise_batch(noise_factor=0.5,train_images = train_images_orginal_R256,test_images = test_images_orginal_R256)

# ************************************************* augmented data set L1000 ***********************************************#

#Add noise to the augmented dataset 1000 resized (128x128)
train_noisy_augmented_L1000_R128, test_noisy_augmented_L1000_R128 = create_data_set_noise_batch(noise_factor=0.5,train_images = train_images_augmented_L1000_R128 ,test_images = test_images_augmented_L1000_R128,batch_size=500)
#Add noise to the augmented dataset 1000 resized (256x256)
train_noisy_augmented_L1000_R256, test_noisy_augmented_L1000_R256 = create_data_set_noise_batch(noise_factor=0.5,train_images = train_images_augmented_L1000_R256,test_images = test_images_augmented_L1000_R256,batch_size=500)

# ************************************************* augmented data set L5000 ***********************************************#
#Add noise to the augmented dataset 5000 resized (128x128)
train_noisy_augmented_L5000_R128, test_noisy_augmented_L5000_R128 = create_data_set_noise_batch(noise_factor=0.5,train_images = train_images_augmented_L5000_R128 ,test_images = test_images_augmented_L5000_R128,batch_size=250)
#Add noise to the augmented dataset 1000 resized (256x256)
train_noisy_augmented_L5000_R256, test_noisy_augmented_L5000_R256 = create_data_set_noise_batch(noise_factor=0.5,train_images = train_images_augmented_L5000_R256,test_images = test_images_augmented_L5000_R256,batch_size=250)
# ************************************************* augmented data set L10000 ***********************************************#
#Add noise to the augmented dataset 5000 resized (128x128)
train_noisy_augmented_L10000_R128, test_noisy_augmented_L10000_R128 = create_data_set_noise_batch(noise_factor=0.5,train_images = train_images_augmented_L10000_R128 ,test_images = test_images_augmented_L10000_R128,batch_size=250)
#Add noise to the augmented dataset 1000 resized (256x256)
train_noisy_augmented_L10000_R256, test_noisy_augmented_L10000_R256 = create_data_set_noise_batch(noise_factor=0.5,train_images = train_images_augmented_L10000_R256,test_images = test_images_augmented_L10000_R256,batch_size=250)

train_noisy_orginal_R128.shape

# Configuration des dimensions des images
IMG_SIZE = 128                        # Taille des images (128x128)
IMG_CHANNELS = 3                      # Canaux des images (RVB)
NB_EPOCHS_DENOISE = 100               # nombre epoch alogithme debruiter
BATCH_SIZE        = 64          # taille batch de traitement
SAV_MODEL_DENOISE = "denoiser.h5"     # sauvegarde du modele de debruitage
latent_dim = 96                  #latent space dimension
# Configuration de l'encodeur (input = image de 128x128 avec 3 canaux)
input_img = Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS))
input_shape = (128, 128, 3)


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
import keras_tuner as kt

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
import keras_tuner as kt
from tensorflow.keras.applications import VGG19

# Perte SSIM
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Perte Perceptuelle basée sur VGG19
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
vgg.trainable = False
layer_outputs = [vgg.get_layer('block3_conv3').output]
feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=layer_outputs)

def perceptual_loss(y_true, y_pred):
    true_features = feature_extractor(y_true)
    pred_features = feature_extractor(y_pred)
    return tf.reduce_mean(tf.square(true_features - pred_features))

# Perte de Variation Totale (TV)
def total_variation_loss(y_true, y_pred):
    return tf.reduce_sum(tf.image.total_variation(y_pred))

# Combinaison de pertes avec SSIM et Total Variation
def combined_loss_with_ssim_tv(y_true, y_pred):
    ssim_l = ssim_loss(y_true, y_pred)
    tv_l = total_variation_loss(y_true, y_pred)
    return 0.9 * ssim_l + 0.1 * tv_l

# Fonction de perte personnalisée RMSE
def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# Taille d'image
IMG_SIZE = 128
IMG_CHANNELS = 3

# Fonction pour construire l'autoencodeur avec Hyperparameter Tuning
def build_autoencoder_model(hp):
    latent_dim = hp.Int('latent_dim', min_value=8, max_value=128, step=8)  # Recherche du latent space
    
    # Optimizer
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    
    # Fonction de perte (mapper les noms à la fonction réelle)
    loss_choice = hp.Choice('loss', ['ssim_loss', 'perceptual_loss', 'total_variation_loss', 'combined_loss_with_ssim_tv'])

    if loss_choice == 'ssim_loss':
        loss = ssim_loss
    elif loss_choice == 'perceptual_loss':
        loss = perceptual_loss
    elif loss_choice == 'total_variation_loss':
        loss = total_variation_loss
    elif loss_choice == 'combined_loss_with_ssim_tv':
        loss = combined_loss_with_ssim_tv

    # Encoder
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS))

    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 128x128 -> 64x64

    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 64x64 -> 32x32

    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)  # 32x32 -> 16x16

    # Latent space
    x = Flatten()(encoded)
    latent = Dense(latent_dim, activation='relu')(x)
    latent = Dropout(0.4)(latent)

    # Decoder
    x = Dense(16 * 16 * 128, activation='relu')(latent)
    x = Reshape((16, 16, 128))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)  # 16x16 -> 32x32

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)  # 32x32 -> 64x64

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)  # 64x64 -> 128x128

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Reconstruction finale

    # Construire le modèle
    autoencoder = Model(input_img, decoded)

    # Compiler le modèle avec les hyperparamètres
    autoencoder.compile(optimizer=optimizer, loss=loss)
    
    return autoencoder


from keras_tuner import RandomSearch

# Instanciation du tuner avec Random Search
tuner = RandomSearch(
    build_autoencoder_model,
    objective='val_loss',  # Critère à optimiser
    max_trials=10,  # Nombre d'essais
    executions_per_trial=1,  # Nombre d'entraînements par combinaison
    directory='autoencoder_tuning',
    project_name='denoising_autoencoder'
)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
import keras_tuner as kt
from tensorflow.keras.applications import VGG19

# Perte SSIM
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Perte Perceptuelle basée sur VGG19
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
vgg.trainable = False
layer_outputs = [vgg.get_layer('block3_conv3').output]
feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=layer_outputs)

def perceptual_loss(y_true, y_pred):
    true_features = feature_extractor(y_true)
    pred_features = feature_extractor(y_pred)
    return tf.reduce_mean(tf.square(true_features - pred_features))

# Perte de Variation Totale (TV)
def total_variation_loss(y_true, y_pred):
    return tf.reduce_sum(tf.image.total_variation(y_pred))

# Combinaison de pertes avec SSIM et Total Variation
def combined_loss_with_ssim_tv(y_true, y_pred):
    ssim_l = ssim_loss(y_true, y_pred)
    tv_l = total_variation_loss(y_true, y_pred)
    return 0.9 * ssim_l + 0.1 * tv_l

# Fonction de perte personnalisée RMSE
def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# Taille d'image
IMG_SIZE = 128
IMG_CHANNELS = 3

# Fonction pour construire l'autoencodeur avec Hyperparameter Tuning
def build_autoencoder_model(hp):
    latent_dim = hp.Int('latent_dim', min_value=8, max_value=128, step=8)  # Recherche du latent space
    
    # Optimizer
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    
    # Fonction de perte (mapper les noms à la fonction réelle)
    loss_choice = hp.Choice('loss', ['ssim_loss', 'perceptual_loss', 'total_variation_loss', 'combined_loss_with_ssim_tv'])

    if loss_choice == 'ssim_loss':
        loss = ssim_loss
    elif loss_choice == 'perceptual_loss':
        loss = perceptual_loss
    elif loss_choice == 'total_variation_loss':
        loss = total_variation_loss
    elif loss_choice == 'combined_loss_with_ssim_tv':
        loss = combined_loss_with_ssim_tv

    # Encoder
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS))

    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 128x128 -> 64x64

    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 64x64 -> 32x32

    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)  # 32x32 -> 16x16

    # Latent space
    x = Flatten()(encoded)
    latent = Dense(latent_dim, activation='relu')(x)
    latent = Dropout(0.4)(latent)

    # Decoder
    x = Dense(16 * 16 * 128, activation='relu')(latent)
    x = Reshape((16, 16, 128))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)  # 16x16 -> 32x32

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)  # 32x32 -> 64x64

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)  # 64x64 -> 128x128

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Reconstruction finale

    # Construire le modèle
    autoencoder = Model(input_img, decoded)

    # Compiler le modèle avec les hyperparamètres
    autoencoder.compile(optimizer=optimizer, loss=loss)
    
    return autoencoder

from keras_tuner import RandomSearch

# Instanciation du tuner avec Random Search
tuner = RandomSearch(
    build_autoencoder_model,
    objective='val_loss',  # Critère à optimiser
    max_trials=10,  # Nombre d'essais
    executions_per_trial=1,  # Nombre d'entraînements par combinaison
    directory='autoencoder_tuning',
    project_name='denoising_autoencoder'
) 

# Lancer une nouvelle recherche d'hyperparamètres avec des fonctions de perte plus adaptées
tuner.search(train_noisy_augmented_L1000_R128, train_images_augmented_L1000_R128, 
             epochs=10,  # Ajustez le nombre d'epochs
             validation_data=(test_noisy_augmented_L1000_R128, test_images_augmented_L1000_R128))

# Récupérer les meilleurs hyperparamètres
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Meilleure dimension du latent space : {best_hps.get('latent_dim')}")
print(f"Meilleur optimiseur : {best_hps.get('optimizer')}")
print(f"Meilleure fonction de perte : {best_hps.get('loss')}")


from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape, Dropout, BatchNormalization
# Encoder
input_img = tf.keras.Input(shape=input_shape)
x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(input_img)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu')(x)  # Réduction 128x128 -> 64x64
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu')(x)  # Réduction 64x64 -> 32x32
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu')(x)  # Réduction 32x32 -> 16x16
x = MaxPooling2D((2, 2), padding='same')(x)  # 128x128 -> 64x64

x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
encoded = Conv2D(512, (3, 3), activation='relu')(x)  # Réduction 16x16 -> 8x8
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Modèle de l'encodeur
encoder_model_R128 = Model(inputs=input_img, outputs=encoded)

# Afficher le résumé de l'encodeur
encoder_model_R128.summary()

# Latent space
# x = Flatten()(encoded)
#latent = Dense(96, activation='relu')(x)
#latent = Dropout(0.4)(latent)

# Optimiseur RMSprop 
optimizer = RMSprop()

# Fonction de perte MSE 
loss = ssim_loss

# Decoder
x = Conv2D(512, (3, 3), activation='relu', padding='same')(encoded)  # Reprend la sortie de l'encodeur
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)  # Agrandir 8x8 -> 16x16

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)  # Agrandir 16x16 -> 32x32

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)  # Agrandir 32x32 -> 64x64

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)  # Agrandir 64x64 -> 128x128

# Reconstruction finale
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Reconstruction finale en 128x128x3

# Modèle du décodeur
decoder_model_R128 = Model(inputs=encoded, outputs=decoded)

# Construire le modèle autoencodeur
autoencoder = Model(input_img, decoded)

# Compiler le modèle avec les meilleurs hyperparamètres
autoencoder.compile(optimizer=optimizer, loss=loss)

# Affichage du résumé du modèle
autoencoder.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history_R128 = autoencoder.fit(
    train_noisy_augmented_L1000_R128, train_images_augmented_L1000_R128,     # Données d'entrée bruitées et images originales correspondantes
    epochs=NB_EPOCHS_DENOISE,   # Nombre d'epochs défini précédemment
    batch_size=BATCH_SIZE,      # Taille de batch définie précédemment
    shuffle=True,               # Shuffle des données à chaque epoch
    validation_data=(test_noisy_augmented_L1000_R128, test_images_augmented_L1000_R128),  # Validation avec les images de test bruitées et originales
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False), early_stopping]  # Une seule liste pour les callbacks
)

autoencoder.save_weights('autoencoder_L1000_R128.weights.h5')

# Visualisation des pertes d'apprentissage (Train) et de validation (Test)
plt.plot(history_R128.history['loss'], label='train')        # Pertes d'entraînement
plt.plot(history_R128.history['val_loss'], label='test')     # Pertes de validation
plt.legend("autoencodeur R128")
plt.title('Courbe d\'apprentissage')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Prédiction des images débruitées avec l'autoencodeur
decoded_imgs = autoencoder.predict(test_noisy_augmented_L1000_R128)

# Fonction pour normaliser les images pour l'affichage
def normalize_image(image):
    """
    Normalise les images à une plage de 0 à 1 pour une meilleure visualisation.
    """
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normaliser l'image
    return image.astype(np.float32)  # Assurez-vous que l'image est en float32

# Nombre d'images à afficher
n = 10

# Initialisation de la figure pour l'affichage
plt.figure(figsize=(20, 6))

# Boucle pour afficher les images bruitées, débruitées et originales
for i in range(n):
    # Affichage des images originales
    ax = plt.subplot(3, n, i  + 1)
    plt.imshow(normalize_image(test_images_augmented_L1000_R128[i]))
    plt.title("Original Image")
    plt.axis("off")

    # Affichage des images bruitées
    ax = plt.subplot(3, n, i+n + 1)
    plt.imshow(normalize_image(test_noisy_augmented_L1000_R128[i]))
    plt.title("Noisy Image")
    plt.axis("off")

    # Affichage des images débruitées
    ax = plt.subplot(3, n, i +2* n + 1)
    plt.imshow(normalize_image(decoded_imgs[i]))
    plt.title("Denoised Image")
    plt.axis("off")



# Afficher la figure avec toutes les images
plt.show()
