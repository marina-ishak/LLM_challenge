import random, os, json, time
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed: int = 1337):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class Timer:
    def __init__(self):
        self.t0 = time.time()
    def lap(self):
        return time.time() - self.t0

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_history(history, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

def plot_confusion_matrix(cm, classes, outpath, normalize=False):
    import itertools
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = "Normalized Confusion Matrix"
    else:
        fmt = 'd'
        title = "Confusion Matrix"
    
    im = ax.imshow(cm, interpolation="nearest", cmap='viridis')
    ax.set_title(title)
    plt.colorbar(im)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
