import tensorflow as tf
from tensorflow.experimental import numpy as tnp


from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM, Dropout, 
    Concatenate, GlobalAveragePooling2D, RepeatVector, 
    Attention, Add, Reshape, Multiply
)
import numpy as np
import json
import os
print(">>> LOADED caption_model.py from:", __file__)

class CaptionModel:
    def __init__(self):
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.max_length = 40  # Maximum caption length
        self.vocab_size = None
        self.embedding_dim = 128
        self.units = 256
        
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
        print(">>> USING NEW build_model (no BroadcastTo)")

        inception = InceptionV3(include_top=False, weights='imagenet')
        inception.trainable = False

        image_input   = Input((299,299,3), name='image_input')
        caption_input = Input((None,),    name='caption_input')

        x = inception(image_input)
        x = GlobalAveragePooling2D()(x)
        image_feats = Dense(self.embedding_dim, activation='relu')(x)

        # replicate image features across time‑steps
        image_seq   = RepeatVector(self.max_length)(image_feats)   # shape=(None,max_len,embed_dim)
        text_embed = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim
        )(caption_input)


        decoder_in  = Concatenate()([image_seq, text_embed])      # (None,max_len,2*embed_dim)
        dec_out     = LSTM(self.units, return_sequences=True)(decoder_in)
        output      = Dense(self.vocab_size)(dec_out)

        self.model = Model([image_input, caption_input], output)
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()

    
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
        for i in range(self.max_length - 1):  # -1 to account for start token
            # Run the model to get predictions
            predictions = self.model.predict([
                np.expand_dims(image, 0),
                decoder_input
            ], verbose=0)
            
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

    def train(self, dataset, val_dataset=None, epochs=20):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        # Train the model
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'model_checkpoint.h5',
                save_best_only=True,
                monitor='val_loss' if val_dataset else 'loss'
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=3,
                monitor='val_loss' if val_dataset else 'loss'
            )
        ]
        
        return self.model.fit(
            dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )