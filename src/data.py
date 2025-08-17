from pathlib import Path
import zipfile, io, requests, os, glob, re
from typing import Tuple, List
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

FSDD_ZIP_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"

def download_fsdd(target_dir: str = "data/fsdd") -> Path:
    target = Path(target_dir).absolute()
    rec = target / "recordings"
    if rec.exists() and any(rec.glob("*.wav")):
        return rec
    target.mkdir(parents=True, exist_ok=True)
    print("Downloading FSDD from GitHub...")
    r = requests.get(FSDD_ZIP_URL, stream=True, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(target)
    # recordings path inside the archive
    rec = target / "free-spoken-digit-dataset-master" / "recordings"
    assert rec.exists(), "Recordings folder not found after extraction."
    # optionally move files one level up
    final = target / "recordings"
    final.mkdir(exist_ok=True)
    for wav in rec.glob("*.wav"):
        wav.replace(final / wav.name)
    return final

DIGIT_PATTERN = re.compile(r"^(\d)_\w+_\d+\.wav$")

def load_fsdd(split=(0.7, 0.15, 0.15), seed=1337) -> Tuple[List[str], List[int], List[str]]:
    rec = download_fsdd()
    files = sorted([str(p) for p in Path(rec).glob("*.wav")])
    y = []
    for f in files:
        m = DIGIT_PATTERN.match(Path(f).name)
        if not m:
            continue
        y.append(int(m.group(1)))
    X = files[:len(y)]
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=1 - split[0], random_state=seed, stratify=y)
    rel = split[1] / (split[1] + split[2])
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=1 - rel, random_state=seed, stratify=y_tmp)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def read_wav(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr
