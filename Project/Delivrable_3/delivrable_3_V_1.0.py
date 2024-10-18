import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

base_dir = os.getcwd()
relative_path = r"..\..\DataSets\Dataset_delivrable_3"
dataset_dir = os.path.normpath(os.path.join(base_dir, relative_path))
#path file data set COCO
train_images_dir = r"C:\Users\steve\Documents\GitHub\DataSciencePictureDesc\DataSets\Dataset_delivrable_3\val2014"
val_images_dir = r"C:\Users\steve\Documents\GitHub\DataSciencePictureDesc\DataSets\Dataset_delivrable_3\val2014"
train_annotations_file = r"C:\Users\steve\Documents\GitHub\DataSciencePictureDesc\DataSets\Dataset_delivrable_3\annotations"

# Initialiser l'objet COCO
coco = COCO(train_annotations_file)

# Récupérer toutes les ID d'images
image_ids = coco.getImgIds()

# Charger une image et ses annotations
def load_image_and_caption(image_id):
    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(train_images_dir, img_info['file_name'])
    img = load_img(img_path, target_size=(224, 224))  # Redimensionner l'image pour ResNet50
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet.preprocess_input(img)  # Prétraitement pour ResNet50
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    captions = [ann['caption'] for ann in anns]
    return img, captions

# Créer le modèle CNN pour l'extraction des caractéristiques (ResNet50)
def create_cnn_model():
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)  # Dernière couche avant la classification
    return model

# Fonction pour extraire les caractéristiques d'images à partir de ResNet50
def extract_image_features(image_ids, coco, resnet_model):
    features = []
    for img_id in image_ids:
        img, _ = load_image_and_caption(img_id)  # Charger l'image
        img_features = resnet_model.predict(img)  # Extraire les caractéristiques avec ResNet50
        features.append(np.squeeze(img_features))
    return np.array(features)

# Prétraiter les légendes
def preprocess_captions(captions, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

# Modèle pour la partie RNN (LSTM pour le texte)
def create_captioning_model(vocab_size, max_sequence_length):
    # Entrée des caractéristiques d'image (sortie du CNN)
    image_input = Input(shape=(2048,))

    # Ajouter une couche dense pour transformer les caractéristiques en vecteur de taille fixe
    image_dense = Dense(256, activation='relu')(image_input)

    # Entrée de la légende
    text_input = Input(shape=(max_sequence_length,))
    text_embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(text_input)
    text_lstm = LSTM(256)(text_embedding)

    # Ajouter et fusionner les deux parties (image + légende)
    decoder = Add()([image_dense, text_lstm])
    decoder = Dense(vocab_size, activation='softmax')(decoder)

    # Créer le modèle
    model = Model(inputs=[image_input, text_input], outputs=decoder)
    return model

# Exemple de vocabulaire et longueur maximale de séquence
vocab_size = 5000
max_sequence_length = 20

# Créer le modèle CNN-RNN
captioning_model = create_captioning_model(vocab_size, max_sequence_length)

# Compiler le modèle
captioning_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Créer le modèle ResNet50 pour l'extraction des caractéristiques
resnet_model = create_cnn_model()

# Exemple d'extraction d'images et légendes
image_ids_subset = image_ids[:1000]  # On peut limiter à un sous-ensemble pour l'exemple
image_features = extract_image_features(image_ids_subset, coco, resnet_model)

# Traiter les légendes
all_captions = []
for img_id in image_ids_subset:
    _, captions = load_image_and_caption(img_id)
    all_captions.extend(captions)

# Créer le tokenizer à partir de toutes les légendes
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(all_captions)

# Séquences de légendes traitées
sequences = [preprocess_captions([caption], tokenizer, max_sequence_length) for caption in all_captions]

# On prend une légende par image pour l'entraînement (par simplicité)
train_captions = [seq[0] for seq in sequences[:1000]]

# Convertir les légendes en one-hot encoding pour l'entraînement
train_captions_encoded = to_categorical(train_captions, num_classes=vocab_size)

# Entraîner le modèle
captioning_model.fit([image_features, np.array(train_captions)], train_captions_encoded, epochs=5)

# Visualisation des pertes d'apprentissage (exemple)
history = captioning_model.fit([image_features, np.array(train_captions)], train_captions_encoded, epochs=5)

# Afficher la courbe de perte
plt.plot(history.history['loss'], label='train loss')
plt.title('Courbe de perte pendant l\'entraînement')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()
plt.show()
