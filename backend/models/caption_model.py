import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM,
    Concatenate, GlobalAveragePooling2D, RepeatVector
)
from tensorflow.keras.applications import MobileNetV2

# Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class CaptionModel:
    def __init__(self):
        self.model        = None
        self.max_length   = 30      # shortened for speed
        self.embedding_dim= 64      # reduced embedding
        self.units        = 128     # smaller LSTM
        
        vocab_path = os.path.join(os.path.dirname(__file__), 'vocabulary.json')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                data = json.load(f)
            self.word_to_index = data['word_to_index']
            self.index_to_word = data['index_to_word']
            self.vocab_size     = len(self.word_to_index)
    
    def build_model(self):
        print(">>> USING MobileNetV2 backbone and automatic masking")
        
        base = MobileNetV2(input_shape=(160,160,3), include_top=False, weights='imagenet')
        base.trainable = False
        
        img_in = Input((160,160,3), name='image_input')
        cap_in = Input((None,),    name='caption_input')
        
        # Image feature extraction
        x = base(img_in)
        x = GlobalAveragePooling2D()(x)
        img_feats = Dense(self.embedding_dim, activation='relu')(x)
        
        # Repeat image features to match caption sequence length
        img_seq = RepeatVector(self.max_length)(img_feats)
        
        # Text embedding with masking enabled
        text_emb = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True  # automatic padding mask
        )(cap_in)
        
        # Combine image features with text embeddings
        decoder_in = Concatenate()([img_seq, text_emb])
        
        # LSTM decoder - will automatically use the mask from text_emb
        dec_out = LSTM(self.units, return_sequences=True)(decoder_in)
        outputs = Dense(self.vocab_size)(dec_out)
        
        self.model = Model([img_in, cap_in], outputs)
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.summary()
    
    def train(self, train_ds, val_ds=None, epochs=10):
        lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
        )
        es_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3
        )
        return self.model.fit(
            train_ds, validation_data=val_ds,
            epochs=epochs, callbacks=[lr_cb, es_cb]
        )