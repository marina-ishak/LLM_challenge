import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

def pre_emphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter to boost higher frequencies"""
    return np.append(x[0], x[1:] - coeff * x[:-1])

def vad_trim(x: np.ndarray, sr: int, top_db: int = 25) -> np.ndarray:
    """Voice Activity Detection - trim silence from the beginning and end"""
    y, _ = librosa.effects.trim(x, top_db=top_db)
    return y if y.size > 0 else x

def normalize_audio(x: np.ndarray) -> np.ndarray:
    """Normalize audio to have zero mean and unit variance"""
    if x.std() > 0:
        return (x - x.mean()) / (x.std() + 1e-10)
    return x

def extract_logmel(x: np.ndarray, sr: int, n_mels: int = 40, n_fft: int = 512, 
                  hop_ms: int = 10, win_ms: int = 25) -> np.ndarray:
    """Extract log mel spectrogram features from audio"""
    # Normalize audio before feature extraction
    x = normalize_audio(x)
    
    hop = int(sr * hop_ms / 1000)
    win = int(sr * win_ms / 1000)
    
    # Use a larger window for better frequency resolution
    S = librosa.feature.melspectrogram(
        y=x, sr=sr, n_fft=n_fft, 
        hop_length=hop, win_length=win, 
        n_mels=n_mels, power=2.0
    )
    
    # Add small epsilon to avoid log(0)
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80)
    
    # Normalize the spectrogram
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-10)
    
    return S_db

def extract_mfcc(x: np.ndarray, sr: int, n_mfcc: int = 20, n_mels: int = 40) -> np.ndarray:
    """Extract MFCC features from audio"""
    # Normalize audio
    x = normalize_audio(x)
    
    # Extract mel spectrogram
    S = extract_logmel(x, sr, n_mels=n_mels)
    
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(S=librosa.db_to_power(S), n_mfcc=n_mfcc, sr=sr)
    
    # Normalize MFCCs
    mfccs = (mfccs - mfccs.mean(axis=1, keepdims=True)) / (mfccs.std(axis=1, keepdims=True) + 1e-10)
    
    return mfccs

def add_deltas(F: np.ndarray) -> np.ndarray:
    """Add first and second order deltas to features"""
    d1 = librosa.feature.delta(F)
    d2 = librosa.feature.delta(F, order=2)
    return np.vstack([F, d1, d2])

def pad_or_crop(F: np.ndarray, max_frames: int = 80) -> np.ndarray:
    """Pad or crop feature matrix to a fixed length"""
    # F: [feature_dim, time]
    T = F.shape[1]
    if T == max_frames:
        return F
    if T < max_frames:
        pad = max_frames - T
        return np.pad(F, ((0, 0), (0, pad)), mode="constant")
    # crop center
    start = (T - max_frames) // 2
    return F[:, start:start+max_frames]

def simple_augment(x: np.ndarray, sr: int) -> np.ndarray:
    """Apply simple audio augmentations"""
    # time shift
    shift = np.random.randint(-int(0.04 * sr), int(0.04 * sr))
    x = np.roll(x, shift)
    
    # Add more aggressive noise for better robustness
    noise_level = np.random.uniform(0.002, 0.008)
    noise = np.random.randn(len(x)).astype(np.float32) * noise_level
    x = x + noise
    
    # Random pitch shift (small amount)
    if np.random.random() > 0.5:
        steps = np.random.uniform(-1, 1)
        x = librosa.effects.pitch_shift(x, sr=sr, n_steps=steps)
        
    # Random speed perturbation (time stretching)
    if np.random.random() > 0.5:
        rate = np.random.uniform(0.9, 1.1)
        x = librosa.effects.time_stretch(x, rate=rate)
        
    return x
