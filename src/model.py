from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def tiny_cnn(input_shape: Tuple[int, int, int] = (60, 80, 1), num_classes: int = 10):
    """
    Improved CNN model for audio classification with more layers and regularization
    """
    inp = keras.Input(shape=input_shape)
    
    # First block
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.1)(x)
    
    # Second block
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Third block
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    
    # Fully connected layers
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
    
    # Output layer
    out = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inp, out)
    return model

def simple_rnn_model(input_shape: Tuple[int, int, int] = (60, 80, 1), num_classes: int = 10):
    """
    Alternative RNN model for audio classification
    """
    # Reshape for RNN: (batch, freq, time, channels) -> (batch, time, freq*channels)
    inp = keras.Input(shape=input_shape)
    reshaped = layers.Reshape((input_shape[1], input_shape[0] * input_shape[2]))(inp)
    
    # RNN layers
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(reshaped)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inp, out)
    return model
