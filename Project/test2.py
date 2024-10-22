from pycocotools.coco import COCO
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import collections
import numpy as np
import os
import time
import json
from tqdm import tqdm
from PIL import Image

# Définir le répertoire de base
base_dir = os.getcwd()
relative_path = r"../../DataSets/Dataset_delivrable_3"
dataset_dir = os.path.normpath(os.path.join(base_dir, relative_path))

# Chemin du fichier d'annotations
annotation_file = os.path.join(dataset_dir, 'annotations/captions_train2014.json')

# Chemin du dossier contenant les images à annoter
image_folder = os.path.join(dataset_dir, 'train2014/')
PATH = image_folder

# Affichage des chemins
print("Dossier des images d'entraînement:", image_folder)
print("Fichier des annotations d'entraînement:", annotation_file)

# Lecture du fichier d'annotation
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# --- Lecture des Annotations ---
image_path_to_caption = collections.defaultdict(list)
for val in annotations['annotations']:
    caption = val['caption']
    image_path = os.path.join(PATH, f"COCO_train2014_{val['image_id']:012d}.jpg")
    image_path_to_caption[image_path].append(caption)

# Vérifiez le contenu de image_path_to_caption
print(f"Nombre d'images avec annotations: {len(image_path_to_caption)}")

# Limiter à 6000 images pour accélérer le prétraitement et l'entraînement
train_image_paths = list(image_path_to_caption.keys())[:2000]

print(f"Nombre d'images sélectionnées pour l'entraînement : {len(train_image_paths)}")

# --- Collecte des légendes et des images associées ---
train_captions = []
img_name_vector = []

for image_path in train_image_paths:
    caption_list = image_path_to_caption[image_path]
    train_captions.extend(caption_list)
    img_name_vector.extend([image_path] * len(caption_list))  # Répéter le chemin d'image pour chaque légende

# Ajouter des tokens de début et de fin aux captions
train_captions = ['start ' + caption + ' end' for caption in train_captions]

# --- Tokenization des légendes ---
top_k = 5000  # Limite le vocabulaire à 5000 mots les plus fréquents
custom_filters = '!"#$%&()*+,-./:;=?@[\]^_`{|}~\t\n'
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token="<unk>", filters=custom_filters)
tokenizer.fit_on_texts(train_captions)

# Convertir les légendes en séquences d'entiers et les padder
train_seqs = tokenizer.texts_to_sequences(train_captions)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# Calcul de la longueur maximale des séquences
def calc_max_length(sequences):
    return max(len(seq) for seq in sequences)

max_length = calc_max_length(train_seqs)
print(f"Longueur maximale des séquences: {max_length}")

# --- Division des données en entraînement et validation ---
img_name_train, img_name_val, cap_train, cap_val = train_test_split(
    img_name_vector, cap_vector, test_size=0.2, random_state=42)

print(f"Nombre d'échantillons d'entraînement : {len(img_name_train)}")
print(f"Nombre d'échantillons de validation : {len(img_name_val)}")

# --- Prétraitement des images et extraction des features avec InceptionV3 ---
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
features_dir = 'extracted_features/'
os.makedirs(features_dir, exist_ok=True)
image_model.trainable = False  # Geler les couches du modèle

new_input = tf.keras.Input(shape=(299, 299, 3))
hidden_layer = image_model(new_input)
image_features_extract_model = tf.keras.Model(inputs=new_input, outputs=hidden_layer)

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [299, 299])
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# Pré-traitement des images
encode_train = sorted(set(img_name_vector))
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

# Extraction et sauvegarde des features d'image
for img, path in tqdm(image_dataset, desc="Extraction des features d'image"):
    batch_features = image_features_extract_model(img)
    batch_features = tf.reduce_mean(batch_features, axis=[1, 2])  # Global Average Pooling
    
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        file_name = os.path.basename(path_of_feature)
        save_path = os.path.join(features_dir, file_name + "_extract_features.npy")
        np.save(save_path, bf.numpy())

# --- Définition du modèle avec Bahdanau Attention ---
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        score = self.V(attention_hidden_layer)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Définition des paramètres
embedding_dim = 256
units = 512
vocab_size = top_k + 1  # +1 pour le token <unk>
feature_dim = 2048

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# Création du modèle de décodage
caption_model = RNN_Decoder(embedding_dim, units, vocab_size)
caption_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# --- Préparation des datasets ---
def map_func(img_name, cap):
    img_tensor = np.load(os.path.join(features_dir, os.path.basename(img_name.numpy().decode('utf-8')) + "_extract_features.npy"))
    return img_tensor.astype(np.float32), cap

def tf_map_func(img_name, cap):
    img_tensor, cap = tf.py_function(map_func, [img_name, cap], [tf.float32, tf.int32])
    img_tensor.set_shape((feature_dim,))
    cap.set_shape((max_length,))
    return img_tensor, cap

train_dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
train_dataset = train_dataset.map(tf_map_func, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
val_dataset = val_dataset.map(tf_map_func, num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(tf.data.AUTOTUNE)

# --- Boucle d'entraînement ---
EPOCHS = 20
loss_plot = []

for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0
    
    for (batch, (img_tensor, target)) in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        loss = 0
        with tf.GradientTape() as tape:
            hidden = caption_model.reset_state(batch_size=target.shape[0])
            dec_input = tf.expand_dims([tokenizer.word_index['start']] * target.shape[0], 1)
            for i in range(1, target.shape[1]):
                predictions, hidden, _ = caption_model(dec_input, img_tensor, hidden)
                loss += tf.keras.losses.sparse_categorical_crossentropy(target[:, i], predictions, from_logits=True)
                dec_input = tf.expand_dims(target[:, i], 1)

            batch_loss = tf.reduce_mean(loss) / int(target.shape[1])
        
        gradients = tape.gradient(batch_loss, caption_model.trainable_variables)
        caption_model.optimizer.apply_gradients(zip(gradients, caption_model.trainable_variables))
        total_loss += batch_loss

    avg_loss = total_loss / (batch + 1)
    loss_plot.append(avg_loss)
    
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Time: {time.time() - start:.2f} sec')

# --- Visualisation de la courbe de perte ---
plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
