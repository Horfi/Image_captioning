import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM, Dropout, 
    MultiHeadAttention, LayerNormalization, Add
)
import numpy as np
import json
import os

class CaptionModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_length = 40  # Maximum caption length
        self.vocab_size = None
        self.embedding_dim = 256
        self.units = 512
        self.attention_heads = 8
        
        # Load vocabulary if exists
        vocab_path = os.path.join(os.path.dirname(__file__), 'vocabulary.json')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
                self.word_to_index = vocab_data['word_to_index']
                self.index_to_word = vocab_data['index_to_word']
                self.vocab_size = len(self.word_to_index)
                self.start_token = self.word_to_index['<start>']
                self.end_token = self.word_to_index['<end>']
    
    def build_model(self):
        """Build the CNN-Transformer model for image captioning"""
        # Image encoder (CNN)
        encoder = InceptionV3(include_top=False, weights='imagenet')
        encoder_output = encoder.output
        encoder_output = tf.keras.layers.GlobalAveragePooling2D()(encoder_output)
        encoder_output = tf.keras.layers.Dense(self.embedding_dim)(encoder_output)
        
        # Text decoder with Transformer
        decoder_input = Input(shape=(None,))
        decoder_embedding = Embedding(self.vocab_size, self.embedding_dim)(decoder_input)
        
        # Transformer decoder layer
        mha = MultiHeadAttention(num_heads=self.attention_heads, key_dim=self.embedding_dim)
        norm1 = LayerNormalization(epsilon=1e-6)
        norm2 = LayerNormalization(epsilon=1e-6)
        dense1 = Dense(self.units, activation='relu')
        dense2 = Dense(self.embedding_dim)
        
        # Cross-attention with image features
        expanded_encoder_output = tf.expand_dims(encoder_output, 1)  # Add sequence dimension
        attn_output = mha(
            query=decoder_embedding,
            key=expanded_encoder_output,
            value=expanded_encoder_output
        )
        
        # Add & Norm
        attn_output = Add()([attn_output, decoder_embedding])
        attn_output = norm1(attn_output)
        
        # Feed Forward Network
        ffn_output = dense1(attn_output)
        ffn_output = dense2(ffn_output)
        
        # Add & Norm
        decoder_output = Add()([ffn_output, attn_output])
        decoder_output = norm2(decoder_output)
        
        # Final output layer
        outputs = Dense(self.vocab_size)(decoder_output)
        
        # Create the full model
        self.model = Model(inputs=[encoder.input, decoder_input], outputs=outputs)
    
    def load_model(self, weights_path):
        """Load pre-trained model weights"""
        if self.model is None and self.vocab_size is not None:
            self.build_model()
            self.model.load_weights(weights_path)
        else:
            print("Error: Vocabulary must be loaded before model can be built")
    
    def generate_caption(self, image):
        """Generate a caption for the provided image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        # Initialize with start token
        decoder_input = tf.expand_dims([self.start_token], 0)
        result = []
        
        # Generate words until max length or end token
        for i in range(self.max_length):
            predictions = self.model([image[np.newaxis, ...], decoder_input])
            # Get the predicted ID for the next word
            predicted_id = tf.argmax(predictions[0, i, :]).numpy()
            
            # If end token is predicted, stop
            if predicted_id == self.end_token:
                break
            
            # Add the predicted word to the result
            result.append(self.index_to_word.get(str(predicted_id), ''))
            
            # Update the decoder input for the next iteration
            decoder_input = tf.concat([decoder_input, [[predicted_id]]], axis=-1)
        
        # Return the generated caption
        return ' '.join(result)

    def train(self, dataset, epochs=20):
        """
        Train the model (simplified function - in practice would be more complex)
        """
        if self.model is None:
            self.build_model()
        
        # Configure the model for training
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Train the model
        # Note: This is simplified - actual training would involve more complex data handling
        return self.model.fit(dataset, epochs=epochs)