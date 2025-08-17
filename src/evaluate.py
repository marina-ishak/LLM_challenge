import argparse, json, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from .train import dataset_to_arrays
from .data import load_fsdd
from .utils import plot_confusion_matrix
import tensorflow as tf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", choices=["mfcc","logmel"], default="mfcc")
    ap.add_argument("--with-deltas", action="store_true")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    (_, _), (_, _), (Xte, yte) = load_fsdd()
    X_test, y_test = dataset_to_arrays(Xte, yte, args.features, args.with_deltas, augment=False)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    
    # Save confusion matrix
    plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], outpath="artifacts/confusion_matrix_eval.png")
    
    # Print prediction distribution for debugging
    print(f"Prediction distribution: {np.bincount(y_pred, minlength=10)}")
    print(f"True distribution: {np.bincount(y_true, minlength=10)}")
    
    rep = classification_report(y_true, y_pred, digits=4)
    print(rep)
    with open("artifacts/classification_report_eval.txt","w") as f:
        f.write(rep)

if __name__ == "__main__":
    main()
