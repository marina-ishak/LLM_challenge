"""
Simple training script for audio classification model
"""
import sys
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.data import load_fsdd, read_wav
from src.features import pre_emphasis, vad_trim, extract_mfcc, normalize_audio, pad_or_crop
from src.model import tiny_cnn
from src.utils import set_seed, ensure_dir

def main():
    # Set parameters
    features_type = "mfcc"
    with_deltas = False
    batch_size = 16
    epochs = 30
    seed = 1337
    
    # Set seed for reproducibility
    set_seed(seed)
    ensure_dir("artifacts")
    
    print("Loading dataset...")
    (Xtr, ytr), (Xv, yv), (Xte, yte) = load_fsdd(seed=seed)
    
    # Print data distribution
    print(f"Training labels: {np.bincount(ytr)}")
    print(f"Validation labels: {np.bincount(yv)}")
    print(f"Test labels: {np.bincount(yte)}")
    
    # Convert data to arrays
    print("Extracting features...")
    X_train, y_train, X_val, y_val, X_test, y_test = process_data(Xtr, ytr, Xv, yv, Xte, yte)
    
    # Create and train model
    print("Creating and training model...")
    input_shape = X_train.shape[1:]
    print(f"Input shape: {input_shape}")
    
    model = tiny_cnn(input_shape=input_shape, num_classes=10)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint("artifacts/simple_model.keras", 
                                              monitor="val_accuracy", 
                                              save_best_only=True)
        ]
    )
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Print statistics
    print(f"Prediction distribution: {np.bincount(y_pred, minlength=10)}")
    print(f"True distribution: {np.bincount(y_true, minlength=10)}")
    
    # Save classification report
    report = classification_report(y_true, y_pred, digits=4)
    print("\nClassification Report:")
    print(report)
    with open("artifacts/simple_classification_report.txt", "w") as f:
        f.write(report)
        
    print("Training complete!")

def process_data(Xtr, ytr, Xv, yv, Xte, yte):
    # Process training data
    X_train, y_train = [], []
    for path, label in zip(Xtr, ytr):
        try:
            # Load and preprocess audio
            x, sr = read_wav(path)
            x = normalize_audio(pre_emphasis(vad_trim(x, sr)))
            
            # Extract features
            features = extract_mfcc(x, sr)
            features = pad_or_crop(features)
            
            # Ensure features are valid
            if np.isnan(features).any() or np.isinf(features).any():
                print(f"Warning: NaN or Inf in features for {path}")
                continue
                
            X_train.append(features)
            y_train.append(label)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    # Process validation data
    X_val, y_val = [], []
    for path, label in zip(Xv, yv):
        try:
            x, sr = read_wav(path)
            x = normalize_audio(pre_emphasis(vad_trim(x, sr)))
            features = extract_mfcc(x, sr)
            features = pad_or_crop(features)
            
            if np.isnan(features).any() or np.isinf(features).any():
                continue
                
            X_val.append(features)
            y_val.append(label)
        except Exception:
            continue
    
    # Process test data
    X_test, y_test = [], []
    for path, label in zip(Xte, yte):
        try:
            x, sr = read_wav(path)
            x = normalize_audio(pre_emphasis(vad_trim(x, sr)))
            features = extract_mfcc(x, sr)
            features = pad_or_crop(features)
            
            if np.isnan(features).any() or np.isinf(features).any():
                continue
                
            X_test.append(features)
            y_test.append(label)
        except Exception:
            continue
    
    # Convert to numpy arrays and add channel dimension
    X_train = np.array(X_train)[..., np.newaxis]
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    
    X_val = np.array(X_val)[..., np.newaxis]
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)
    
    X_test = np.array(X_test)[..., np.newaxis]
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    
    print(f"Training data: {X_train.shape}, {y_train.shape}")
    print(f"Validation data: {X_val.shape}, {y_val.shape}")
    print(f"Test data: {X_test.shape}, {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    main()
