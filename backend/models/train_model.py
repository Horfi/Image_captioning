import tensorflow as tf
import numpy as np
import os
import json
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_v3 import preprocess_input
from caption_model import CaptionModel


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
        try:
            parts = line.split(',', 1)
            # Make sure to get just the filename without any extra parts
            image_file = parts[0].split('#')[0].strip()
            
            # Skip non-image files
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                continue
                
            # Verify the image file exists
            if not os.path.exists(os.path.join(images_dir, image_file)):
                print(f"Warning: Image file not found: {image_file}")
                continue
            
            caption = parts[1].strip()
            
            if image_file not in image_to_captions:
                image_to_captions[image_file] = []
            image_to_captions[image_file].append(caption)
        except Exception as e:
            print(f"Error processing line: {line}")
            print(f"Error: {e}")
    
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
    
    # Make sure essential tokens are included
    essential_tokens = ['<start>', '<end>', '<pad>', '<unk>']
    for token in essential_tokens:
        if token not in filtered_words:
            filtered_words.append(token)
    
    # Create word-to-index mapping
    word_to_index = {'<pad>': 0, '<unk>': 1}  # Add unknown token
    for i, word in enumerate(filtered_words, 2):  # Start from 2 to account for pad and unk
        if word not in word_to_index:  # Avoid duplicates
            word_to_index[word] = i
    
    # Create index-to-word mapping
    index_to_word = {str(i): word for word, i in word_to_index.items()}
    
    return word_to_index, index_to_word

def create_dataset(image_paths, captions, word_to_index, max_length, batch_size=32):
    """Create a TensorFlow dataset for training with improved padding handling"""
    # Convert file paths and captions to lists
    all_img_paths = []
    all_input_seqs = []
    all_target_seqs = []
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        if img_name not in captions:
            continue
            
        for caption in captions[img_name]:
            # Convert words to indices
            sequence = [word_to_index.get(word, word_to_index.get('<unk>', 1)) for word in caption.split()]
            
            # Skip sequences that are too short
            if len(sequence) < 3:  # At least <start>, one word, and <end>
                continue
                
            # Truncate if necessary
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
            
            # Create input and target sequences (without padding for now)
            input_seq = sequence[:-1]  # all words except the last one
            target_seq = sequence[1:]  # all words except the first one
            
            all_img_paths.append(img_path)
            all_input_seqs.append(input_seq)
            all_target_seqs.append(target_seq)
    
    # Function to pad sequences
    def pad_sequences(sequences, maxlen):
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=maxlen, padding='post', truncating='post')
    
    # Pad all sequences to the same length
# new
    padded_input_seqs  = pad_sequences(all_input_seqs,  max_length)
    padded_target_seqs = pad_sequences(all_target_seqs, max_length)

    
    # Convert to tensors
    input_tensor = tf.convert_to_tensor(padded_input_seqs, dtype=tf.int32)
    target_tensor = tf.convert_to_tensor(padded_target_seqs, dtype=tf.int32)
    
    # Create a dataset
    dataset = tf.data.Dataset.from_tensor_slices((all_img_paths, input_tensor, target_tensor))
    
    # Map function to load images
    def map_func(img_path, input_seq, target_seq):
        # Load and preprocess image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = preprocess_input(img)
        
        return {'image_input': img, 'caption_input': input_seq}, target_seq
    
    # Apply mapping, shuffle, and batch
    dataset = dataset.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(buffer_size=500)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def main():
    # Set memory growth for GPUs if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth set for GPU: {device}")
    
    # Paths - Update these to match your actual directory structure
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'Flickr8k_Dataset')
    captions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'Flickr8k_text', 'captions.txt')
    
    # Make sure the directory exists
    print(f"Checking if directory exists: {images_dir}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Print a few example files to verify the path is correct
    image_files = os.listdir(images_dir)[:5]
    print(f"Example image files: {image_files}")
    
    # Load dataset
    print("Loading dataset...")
    image_to_captions = load_flickr8k_dataset(images_dir, captions_file)
    all_images = list(image_to_captions.keys())[:500]
    image_to_captions = {k: image_to_captions[k] for k in all_images}
    
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
    batch_size = 8  # Reduced batch size to avoid memory issues
    
    # Split dataset
    print("Splitting dataset...")
    # Only include image files in paths
    image_paths = [os.path.join(images_dir, img) for img in image_to_captions.keys() 
                if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
    
    # Create datasets
    print("Creating training dataset...")
    train_dataset = create_dataset(
        train_paths, 
        {os.path.basename(p): image_to_captions[os.path.basename(p)] for p in train_paths},
        word_to_index, 
        max_length,
        batch_size
    )
    
    print("Creating validation dataset...")
    val_dataset = create_dataset(
        val_paths, 
        {os.path.basename(p): image_to_captions[os.path.basename(p)] for p in val_paths},
        word_to_index, 
        max_length,
        batch_size
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
    # Train for 10 epochs
    history = model.train(train_dataset, val_dataset, epochs=5)

    print("Training complete!")
    
    keras_path = os.path.join(os.path.dirname(__file__), 'caption_model_500.keras')
    print(f"⏳ Saving Keras model to {keras_path}…")
    model.model.save(keras_path)
    print(f"✅ Saved Keras model to {keras_path}")


if __name__ == "__main__":
    main()