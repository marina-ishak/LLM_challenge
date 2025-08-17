import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Make sure the src directory is in the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import tensorflow as tf
except ImportError:
    print("tensorflow not found. Installing tensorflow...")
    os.system("pip install tensorflow")
    import tensorflow as tf

# Import project modules
from src.data import load_fsdd, read_wav
from src.features import pre_emphasis, vad_trim, extract_mfcc, extract_logmel, add_deltas, pad_or_crop, normalize_audio
from src.train import dataset_to_arrays
from src.model import tiny_cnn

def plot_features(features, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(features, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"artifacts/{title.replace(' ', '_').lower()}.png")
    plt.close()

def main():
    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    
    print("="*50)
    print("DEBUGGING AUDIO CLASSIFICATION MODEL")
    print("="*50)
    
    # 1. Check dataset distribution
    print("\n1. Checking dataset distribution...")
    try:
        (Xtr, ytr), (Xv, yv), (Xte, yte) = load_fsdd()
        
        print(f"Training set: {len(Xtr)} samples")
        print(f"Validation set: {len(Xv)} samples")
        print(f"Test set: {len(Xte)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Count classes in each split
    train_counts = Counter(ytr)
    val_counts = Counter(yv)
    test_counts = Counter(yte)
    
    print("\nClass distribution in training set:")
    for i in range(10):
        print(f"  Class {i}: {train_counts[i]} samples")
    
    print("\nClass distribution in test set:")
    for i in range(10):
        print(f"  Class {i}: {test_counts[i]} samples")
    
    # 2. Check feature extraction
    print("\n2. Checking feature extraction...")
    
    # Sample a few files from different classes
    sample_indices = []
    for digit in range(10):
        indices = [i for i, y in enumerate(ytr) if y == digit]
        if indices:
            sample_indices.append(indices[0])
    
    for idx in sample_indices[:3]:  # Take first 3 samples for brevity
        wav_path = Xtr[idx]
        digit = ytr[idx]
        
        print(f"\nExtracting features for sample {Path(wav_path).name} (digit {digit}):")
        
        # Extract raw audio
        x, sr = read_wav(wav_path)
        print(f"  Raw audio shape: {x.shape}, Sample rate: {sr}")
        print(f"  Raw audio range: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}, std={x.std():.4f}")
        
        # Process audio
        x_processed = pre_emphasis(vad_trim(x, sr))
        print(f"  Processed audio shape: {x_processed.shape}")
        print(f"  Processed audio range: min={x_processed.min():.4f}, max={x_processed.max():.4f}")
        
        # Extract features
        mfcc_features = extract_mfcc(x_processed, sr)
        logmel_features = extract_logmel(x_processed, sr)
        
        print(f"  MFCC shape: {mfcc_features.shape}")
        print(f"  LogMel shape: {logmel_features.shape}")
        
        # Add deltas and check
        mfcc_deltas = add_deltas(mfcc_features)
        print(f"  MFCC with deltas shape: {mfcc_deltas.shape}")
        
        # Check after padding/cropping
        mfcc_padded = pad_or_crop(mfcc_features)
        print(f"  Padded MFCC shape: {mfcc_padded.shape}")
        
        # Plot features
        plot_features(mfcc_features, f"MFCC Features - Digit {digit}")
        plot_features(logmel_features, f"LogMel Features - Digit {digit}")
    
    # 3. Check model input
    print("\n3. Checking model input...")
    
    # Create small batch of data
    features = "mfcc"
    with_deltas = False
    X_sample, y_sample = dataset_to_arrays(Xtr[:10], ytr[:10], features, with_deltas)
    
    print(f"Model input shape: {X_sample.shape}")
    print(f"Model input range: min={X_sample.min():.4f}, max={X_sample.max():.4f}, mean={X_sample.mean():.4f}")
    print(f"Labels shape: {y_sample.shape}")
    print(f"Sample labels (one-hot): {y_sample[0]}")
    
    # 4. Check model architecture
    print("\n4. Checking model architecture...")
    input_shape = X_sample.shape[1:]
    model = tiny_cnn(input_shape=input_shape, num_classes=10)
    model.summary()
    
    # 5. Test model predictions before training
    print("\n5. Testing model predictions before training...")
    
    # Make predictions with untrained model
    X_small, y_small = dataset_to_arrays(Xte[:20], yte[:20], features, with_deltas)
    predictions = model.predict(X_small)
    
    print("Untrained model prediction distribution:")
    pred_classes = np.argmax(predictions, axis=1)
    pred_counts = Counter(pred_classes)
    for i in range(10):
        print(f"  Class {i}: {pred_counts[i]} predictions")
    
    # Check prediction probabilities
    print("\nSample prediction probabilities:")
    for i in range(3):
        true_class = np.argmax(y_small[i])
        pred_class = np.argmax(predictions[i])
        print(f"  Sample {i} (true: {true_class}, pred: {pred_class}):")
        for j in range(10):
            print(f"    Class {j}: {predictions[i][j]:.6f}")
    
    print("\n6. Quick training test (overfitting small batch)...")
    # Try to overfit on a very small batch to check if model can learn
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Take just 10 samples and try to overfit
    X_tiny, y_tiny = dataset_to_arrays(Xtr[:10], ytr[:10], features, with_deltas)
    
    # Train for a few epochs
    hist = model.fit(
        X_tiny, y_tiny,
        epochs=20,
        batch_size=5,
        verbose=2
    )
    
    # Check if loss decreased
    print(f"\nInitial loss: {hist.history['loss'][0]:.4f}")
    print(f"Final loss: {hist.history['loss'][-1]:.4f}")
    
    # Check predictions after mini-training
    predictions = model.predict(X_tiny, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_tiny, axis=1)
    
    print("\nAfter mini-training:")
    print(f"  True classes: {true_classes}")
    print(f"  Predicted classes: {pred_classes}")
    
    # Calculate accuracy
    accuracy = np.mean(pred_classes == true_classes)
    print(f"  Accuracy on training data: {accuracy:.4f}")
    
    print("\nDEBUGGING COMPLETE")
    print("="*50)
    print("Check the artifacts directory for feature visualizations")
    print("Based on this debugging, look for:")
    print("1. Issues in feature extraction")
    print("2. Imbalanced dataset")
    print("3. Problems with model initialization or training")
    print("4. Check if model can learn on small batches")

if __name__ == "__main__":
    main()
