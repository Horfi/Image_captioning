import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import tensorflow as tf

def preprocess_image(image, target_size=(299, 299)):
    """
    Preprocess an image for the InceptionV3 model
    
    Args:
        image: PIL Image object
        target_size: tuple of (height, width) for resizing
        
    Returns:
        Preprocessed image as numpy array
    """
    # Resize image to target size
    if image.size != target_size:
        image = image.resize(target_size)
    
    # Convert to RGB if image is in another mode (e.g., RGBA)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and preprocess for InceptionV3
    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    
    return img_array

def preprocess_caption(caption, word_to_index, max_length=40):
    """
    Preprocess a caption for model input
    
    Args:
        caption: String caption
        word_to_index: Dictionary mapping words to indices
        max_length: Maximum length of caption
        
    Returns:
        Tokenized and padded caption as tensor
    """
    # Tokenize
    tokens = caption.lower().split()
    
    # Convert words to indices
    sequence = []
    for word in tokens:
        if word in word_to_index:
            sequence.append(word_to_index[word])
        else:
            # Use an <unk> token or skip
            if '<unk>' in word_to_index:
                sequence.append(word_to_index['<unk>'])
    
    # Pad sequence
    if len(sequence) < max_length:
        sequence = sequence + [0] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    
    return tf.convert_to_tensor([sequence])