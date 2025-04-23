# train_model.py
import os, json, re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from caption_model import CaptionModel

tf.config.optimizer.set_jit(True)  # Enable XLA JIT for faster execution

# 1. Load and preprocess the Flickr8k dataset

def load_flickr8k_dataset(img_dir, cap_file):
    with open(cap_file, 'r') as f:
        lines = f.read().splitlines()
    mapping = {}
    for line in lines:
        if not line: continue
        img, cap = line.split(',', 1)
        img = img.split('#')[0]
        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
            mapping.setdefault(img, []).append(cap.strip())
    return mapping


def preprocess_captions(caps):
    cleaned = []
    for c in caps:
        c = re.sub(r"[^\w\s]", '', c.lower())
        cleaned.append(f"<start> {c} <end>")
    return cleaned

# 2. Tokenizer

def create_tokenizer(mapping):
    counts = {}
    for caps in mapping.values():
        for c in caps:
            for w in c.split(): counts[w] = counts.get(w, 0) + 1
    vocab = [w for w, ct in counts.items() if ct >= 5]
    for tok in ['<start>', '<end>', '<pad>', '<unk>']:
        if tok not in vocab:
            vocab.append(tok)
    word_to_index = {'<pad>': 0, '<unk>': 1}
    for i, w in enumerate(vocab, 2):
        word_to_index[w] = i
    index_to_word = {str(i): w for w, i in word_to_index.items()}
    with open('vocabulary.json', 'w') as vf:
        json.dump({'word_to_index': word_to_index, 'index_to_word': index_to_word}, vf)
    return word_to_index, index_to_word

# 3. Create tf.data dataset

def create_dataset(img_dir, img_paths, mapping, w2i, max_len, batch_size):
    paths, in_seqs, tgt_seqs = [], [], []
    for p in img_paths:
        name = os.path.basename(p)
        for cap in mapping.get(name, []):
            seq = [w2i.get(w, w2i['<unk>']) for w in cap.split()]
            if len(seq) < 3: continue
            if len(seq) > max_len: seq = seq[:max_len]
            in_seqs.append(seq[:-1]); tgt_seqs.append(seq[1:])
            paths.append(p)

    pad = tf.keras.preprocessing.sequence.pad_sequences
    in_pad = pad(in_seqs, maxlen=max_len, padding='post')
    tgt_pad = pad(tgt_seqs, maxlen=max_len, padding='post')

    ds = tf.data.Dataset.from_tensor_slices((paths, in_pad, tgt_pad))
    def map_fn(img_path, inp, tgt):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (160, 160))
        img = preprocess_input(img)
        return {'image_input': img, 'caption_input': inp}, tgt

    return ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE) \
             .cache().shuffle(1000) \
             .batch(batch_size) \
             .prefetch(tf.data.AUTOTUNE)


def main():
    # GPU memory growth
    for dev in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(dev, True)

    base = os.path.abspath(os.path.dirname(__file__))
    img_dir = os.path.join(base, '..', '..', 'data', 'Flickr8k_Dataset')
    cap_file = os.path.join(base, '..', '..', 'data', 'Flickr8k_text', 'captions.txt')

    mapping = load_flickr8k_dataset(img_dir, cap_file)
    for k in mapping: mapping[k] = preprocess_captions(mapping[k])

    w2i, i2w = create_tokenizer(mapping)

    max_len, batch_size = 30, 16
    all_images = [os.path.join(img_dir, fn) for fn in mapping]

    # Sample 30% for quick test
    sample, _ = train_test_split(all_images, train_size=0.3, random_state=42)
    train_paths, val_paths = train_test_split(sample, test_size=0.2, random_state=42)

    train_ds = create_dataset(img_dir, train_paths, mapping, w2i, max_len, batch_size)
    val_ds = create_dataset(img_dir, val_paths, mapping, w2i, max_len, batch_size)

    model = CaptionModel()
    model.vocab_size = len(w2i)
    model.word_to_index = w2i
    model.index_to_word = i2w
    model.start_token = w2i['<start>']
    model.end_token = w2i['<end>']
    model.build_model()

    model.train(train_ds, val_ds, epochs=5)
    model.model.save_weights('model_weights_mobilenetv2.h5')

if __name__ == '__main__':
    main()