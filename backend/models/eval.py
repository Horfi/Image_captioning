import os
import json
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Paths - adjust if needed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, '..', '..', 'data', 'Flickr8k_Dataset')
CAPTIONS_FILE = os.path.join(BASE_DIR, '..', '..', 'data', 'Flickr8k_text', 'captions.txt')
VOCAB_FILE = os.path.join(BASE_DIR, 'vocabulary.json')
SAVED_MODEL = os.path.join(BASE_DIR, 'caption_model_200.keras')  # or .h5/checkpoint path

# Parameters (should match training)
MAX_LENGTH = 40
BATCH_SIZE = 8


def load_flickr8k_dataset(images_dir, captions_file):
    with open(captions_file, 'r') as f:
        lines = f.read().split('\n')
    mapping = {}
    for line in lines:
        if not line.strip(): continue
        img, cap = line.split(',', 1)
        img = img.split('#')[0]
        if not img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            continue
        img_path = os.path.join(images_dir, img)
        if not os.path.exists(img_path):
            continue
        cap = cap.strip()
        mapping.setdefault(img, []).append(cap)
    return mapping


def preprocess_captions(caps):
    processed = []
    for c in caps:
        c = re.sub(r"[^\w\s]", "", c.lower())
        processed.append(f"<start> {c} <end>")
    return processed


def pad_sequences(seqs, maxlen):
    return tf.keras.preprocessing.sequence.pad_sequences(
        seqs, maxlen=maxlen, padding='post', truncating='post')


def create_dataset(img_paths, captions, word_to_index, max_length, batch_size):
    img_list, in_seqs, tgt_seqs = [], [], []
    for path in img_paths:
        name = os.path.basename(path)
        if name not in captions: continue
        for cap in captions[name]:
            seq = [word_to_index.get(w, word_to_index['<unk>']) for w in cap.split()]
            if len(seq) < 3: continue
            if len(seq) > max_length:
                seq = seq[:max_length]
            in_seqs.append(seq[:-1])
            tgt_seqs.append(seq[1:])
            img_list.append(path)
    in_padded = pad_sequences(in_seqs, max_length)
    tgt_padded = pad_sequences(tgt_seqs, max_length)
    ds = tf.data.Dataset.from_tensor_slices((img_list, in_padded, tgt_padded))

    def _map(path, inp, tgt):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = preprocess_input(img)
        return {'image_input': img, 'caption_input': inp}, tgt

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().shuffle(500)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    # Load vocabulary
    with open(VOCAB_FILE, 'r') as f:
        vocab = json.load(f)
    word_to_index = vocab['word_to_index']

    # Load and prepare captions
    data = load_flickr8k_dataset(IMAGES_DIR, CAPTIONS_FILE)
    for k in data:
        data[k] = preprocess_captions(data[k])

    # Split to get validation paths
    all_paths = [os.path.join(IMAGES_DIR, k) for k in data.keys()]
    _, val_paths = train_test_split(all_paths, test_size=0.2, random_state=42)
    val_caps = {os.path.basename(p): data[os.path.basename(p)] for p in val_paths}

    # Build validation dataset
    val_ds = create_dataset(val_paths, val_caps, word_to_index, MAX_LENGTH, BATCH_SIZE)

    # Load model
    model = tf.keras.models.load_model(SAVED_MODEL)
    # Evaluate
    loss, acc = model.evaluate(val_ds)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")


if __name__ == '__main__':
    main()
