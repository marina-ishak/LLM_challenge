import argparse, os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from .data import load_fsdd, read_wav
from .features import pre_emphasis, vad_trim, extract_mfcc, extract_logmel, add_deltas, pad_or_crop, simple_augment, normalize_audio
from .model import tiny_cnn
from .utils import set_seed, ensure_dir, save_history, plot_confusion_matrix

import tensorflow as tf
from tensorflow import keras

def featurize(path, features="mfcc", with_deltas=False, max_frames=80):
    """Extract features from a single audio file"""
    try:
        x, sr = read_wav(path)
        # Apply pre-processing
        x = normalize_audio(pre_emphasis(vad_trim(x, sr)))
        
        if features == "mfcc":
            F = extract_mfcc(x, sr, n_mfcc=20, n_mels=40)
        else:
            F = extract_logmel(x, sr, n_mels=40)
            
        if with_deltas:
            F = add_deltas(F)
            
        F = pad_or_crop(F, max_frames=max_frames)
        return F.astype("float32")
    except Exception as e:
        print(f"Error processing {path}: {e}")
        # Return zeros as fallback
        if with_deltas:
            return np.zeros((60 if with_deltas else 20, max_frames), dtype="float32")
        else:
            return np.zeros((20, max_frames), dtype="float32")

def dataset_to_arrays(paths, labels, features, with_deltas, augment=False):
    """Convert dataset files to feature arrays"""
    X, y = [], []
    
    # Print distribution of labels before processing
    print(f"Label distribution: {np.bincount(labels)}")
    
    for p, lab in zip(paths, labels):
        try:
            if augment:
                # More aggressive augmentation
                x, sr = read_wav(p)
                x = simple_augment(x, sr)
                x = normalize_audio(pre_emphasis(vad_trim(x, sr)))
                
                if features == "mfcc":
                    F = extract_mfcc(x, sr, n_mfcc=20, n_mels=40)
                else:
                    F = extract_logmel(x, sr, n_mels=40)
                    
                if with_deltas:
                    F = add_deltas(F)
                    
                F = pad_or_crop(F, max_frames=80)
            else:
                F = featurize(p, features, with_deltas, max_frames=80)
                
            # Check if features are valid
            if np.isnan(F).any() or np.isinf(F).any():
                print(f"Warning: NaN or Inf in features for {p}")
                continue
                
            X.append(F)
            y.append(lab)
        except Exception as e:
            print(f"Error processing {p}: {e}")
            continue
            
    # Convert to numpy arrays
    X = np.array(X)[..., np.newaxis]  # (N, F, T, 1)
    y = keras.utils.to_categorical(y, num_classes=10)
    
    # Print stats about the processed data
    print(f"Processed {len(X)} samples with shape {X.shape}")
    print(f"Feature stats: min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}, std={X.std():.4f}")
    
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["mfcc","logmel"], default="mfcc")
    ap.add_argument("--with-deltas", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)  # Set to 15 epochs as requested
    ap.add_argument("--batch-size", type=int, default=16)  # Smaller batch size
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--lr", type=float, default=0.001)  # Fixed learning rate
    ap.add_argument("--model-out", default="artifacts/model.keras")
    ap.add_argument("--realtime", action="store_true", help="Optimize model for real-time inference")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir("artifacts")

    print("Loading dataset...")
    (Xtr, ytr), (Xv, yv), (Xte, yte) = load_fsdd(seed=args.seed)
    
    # Print data distribution
    print(f"Training labels: {np.bincount(ytr)}")
    print(f"Validation labels: {np.bincount(yv)}")
    print(f"Test labels: {np.bincount(yte)}")
    
    print("Extracting features...")
    X_train, y_train = dataset_to_arrays(Xtr, ytr, args.features, args.with_deltas, augment=True)
    X_val, y_val = dataset_to_arrays(Xv, yv, args.features, args.with_deltas, augment=False)
    X_test, y_test = dataset_to_arrays(Xte, yte, args.features, args.with_deltas, augment=False)

    # Check for any issues in the data
    if np.isnan(X_train).any() or np.isnan(X_val).any() or np.isnan(X_test).any():
        print("WARNING: NaN values found in the data. Fixing...")
        # Replace NaN with zeros
        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)
        X_test = np.nan_to_num(X_test)

    input_shape = X_train.shape[1:]
    print(f"Input shape: {input_shape}")
    
    # Create and compile model
    print("Creating model...")
    model = tiny_cnn(input_shape=input_shape, num_classes=10)
    
    # Use a simpler learning rate schedule
    opt = keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=opt, 
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )
    
    # Print model summary
    model.summary()

    # Add additional callbacks for better training
    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", 
            patience=10,  # Increased patience
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            args.model_out, 
            monitor="val_accuracy", 
            save_best_only=True, 
            verbose=1
        ),
        # Add learning rate reduction on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Add TensorBoard for visualization
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]

    print(f"Training model for up to {args.epochs} epochs...")
    
    # Compute class weights from y_train
    from sklearn.utils.class_weight import compute_class_weight
    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")
    
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,  # More detailed output
        callbacks=cb,
        class_weight=class_weight_dict  # Use computed class weights
    )

    save_history(hist, "artifacts/history.json")

    # Evaluate on test set
    print("Evaluating model on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")

    # Detailed prediction analysis
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Print prediction distribution
    print(f"Prediction distribution: {np.bincount(y_pred, minlength=10)}")
    print(f"True distribution: {np.bincount(y_true, minlength=10)}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Save confusion matrix
    plot_confusion_matrix(
        cm, 
        classes=[str(i) for i in range(10)], 
        outpath="artifacts/confusion_matrix.png"
    )
    
    # Save classification report
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    with open("artifacts/classification_report.txt", "w") as f:
        f.write(report)

    print("Saved model to", args.model_out)
    print("Training complete!")

if __name__ == "__main__":
    main()
