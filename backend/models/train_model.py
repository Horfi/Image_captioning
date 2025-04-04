import tensorflow as tf
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from caption_model import CaptionModel
import pickle
import re

def load_flickr8k_dataset(images_dir, captions_file):
    """Load the Flickr8k dataset"""
    # Read captions
    with open(captions_file, 'r') as f:
        captions_data = f.read().split('\n')
    
    # Parse captions
    image_to_captions = {}
    for line in captions_data:
        if len(line) < 2:
            continue
        parts = line.split(',', 1)
        image_file = parts[0].split('#')[0].strip()
        caption = parts[1].strip()
        
        if image_file not in image_to_captions:
            image_to_captions[image_file] = []
        image_to_captions[image_file].append(caption)
    
    return image_to_captions

def preprocess_captions(captions):
    """Preprocess captions - lowercase, remove special chars, add start/end tokens"""
    processed_captions = []
    for caption in captions:
        # Convert to lowercase and remove special characters
        caption = re.sub(r'[^\w\s]', '', caption.lower())
        # Add start and end tokens
        caption = f"<start> {caption} <end>"
        processed_captions.append(caption)
    return processed_captions

def create_tokenizer(captions):
    """Create a tokenizer for the captions"""
    # Flatten the list of captions
    all_captions = [cap for caps in captions.values() for cap in caps]
    
    # Build vocabulary
    word_counts = {}
    for caption in all_captions:
        for word in caption.split():
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
    
    # Filter words by frequency if needed
    min_word_count = 5
    filtered_words = [word for word, count in word_counts.items() if count >= min_word_count]
    
    # Create word-to-index mapping
    word_to_index = {'<pad>': 0}
    for i, word in enumerate(filtered_words, 1):
        word_to_index[word] = i
    
    # Create index-to-word mapping
    index_to_word = {str(i): word for word, i in word_to_index.items()}
    
    return word_to_index, index_to_word

def create_dataset(image_paths, captions, word_to_index, max_length, batch_size=32):
    """Create a TensorFlow dataset for training"""
    # Helper function to load and preprocess images
    def load_and_preprocess_image(image_path):
        img = load_img(image_path, target_size=(299, 299))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array
    
    # Convert captions to sequences
    def caption_to_sequence(caption, max_length):
        sequence = [word_to_index.get(word, word_to_index['<pad>']) for word in caption.split()]
        # Pad sequence to max_length
        if len(sequence) < max_length:
            sequence = sequence + [0] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        return sequence
    
    # Prepare data for training
    image_tensors = []
    input_sequences = []
    target_sequences = []
    
    for img_path in image_paths:
        img_tensor = load_and_preprocess_image(img_path)
        for caption in captions[os.path.basename(img_path)]:
            sequence = caption_to_sequence(caption, max_length+1)  # +1 for target shifting
            
            # Input sequence is all words except the last one
            input_seq = sequence[:-1]
            # Target sequence is all words except the first one
            target_seq = sequence[1:]
            
            image_tensors.append(img_tensor)
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)
    
    # Convert to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        (np.array(image_tensors), np.array(input_sequences)), 
        np.array(target_sequences)
    ))
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def main():
    # Paths
    images_dir = '../../data/Flickr8k_Dataset/Flicker8k_Dataset'
    captions_file = '../../data/Flickr8k_text/captions.txt'
    
    # Load dataset
    print("Loading dataset...")
    image_to_captions = load_flickr8k_dataset(images_dir, captions_file)
    
    # Preprocess captions
    print("Preprocessing captions...")
    for img, caps in image_to_captions.items():
        image_to_captions[img] = preprocess_captions(caps)
    
    # Create tokenizer
    print("Creating tokenizer...")
    word_to_index, index_to_word = create_tokenizer(image_to_captions)
    
    # Save vocabulary
    vocab_data = {
        'word_to_index': word_to_index,
        'index_to_word': index_to_word
    }
    with open('vocabulary.json', 'w') as f:
        json.dump(vocab_data, f)
    
    # Parameters
    max_length = 40  # Maximum caption length
    vocab_size = len(word_to_index)
    
    # Split dataset
    print("Splitting dataset...")
    image_paths = [os.path.join(images_dir, img) for img in image_to_captions.keys()]
    train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
    
    # Create datasets
    print("Creating training dataset...")
    train_dataset = create_dataset(
        train_paths, 
        {os.path.basename(p): image_to_captions[os.path.basename(p)] for p in train_paths},
        word_to_index, 
        max_length
    )
    
    print("Creating validation dataset...")
    val_dataset = create_dataset(
        val_paths, 
        {os.path.basename(p): image_to_captions[os.path.basename(p)] for p in val_paths},
        word_to_index, 
        max_length
    )
    
    # Initialize and train model
    print("Initializing model...")
    model = CaptionModel()
    model.vocab_size = vocab_size
    model.word_to_index = word_to_index
    model.index_to_word = index_to_word
    model.start_token = word_to_index['<start>']
    model.end_token = word_to_index['<end>']
    model.build_model()
    
    print("Training model...")
    # Train for fewer epochs initially for testing
    model.train(train_dataset, epochs=10)
    
    # Save model weights
    print("Saving model...")
    model.model.save_weights('model_weights.h5')
    print("Training complete!")

if __name__ == "__main__":
    main()