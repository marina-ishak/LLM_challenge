"""
Debugging and visualization helper for audio classification
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Make sure the src directory is in the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import librosa.display
except ImportError:
    print("librosa.display not found. Installing librosa...")
    os.system("pip install librosa")
    import librosa.display

# Import project modules
from src.data import load_fsdd, read_wav
from src.features import pre_emphasis, vad_trim, extract_mfcc, extract_logmel, normalize_audio
from src.utils import ensure_dir

def visualize_audio(audio_path, output_dir='artifacts/debug'):
    """Visualize audio waveform and its spectrogram"""
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename
    filename = Path(audio_path).stem
    digit = filename.split('_')[0]  # Extract digit from filename
    
    # Load audio
    x, sr = read_wav(audio_path)
    
    # Create a figure with 4 subplots
    plt.figure(figsize=(16, 12))
    
    # 1. Raw waveform
    plt.subplot(4, 1, 1)
    plt.title(f"Raw waveform - Digit {digit}")
    plt.plot(x)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    
    # 2. Processed waveform (pre-emphasis + VAD)
    processed = pre_emphasis(vad_trim(x, sr))
    plt.subplot(4, 1, 2)
    plt.title(f"Processed waveform (pre-emphasis + VAD)")
    plt.plot(processed)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    
    # 3. Mel spectrogram
    plt.subplot(4, 1, 3)
    mel_spec = extract_logmel(normalize_audio(processed), sr)
    librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel', sr=sr)
    plt.title('Mel spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    # 4. MFCC
    plt.subplot(4, 1, 4)
    mfcc = extract_mfcc(normalize_audio(processed), sr)
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.title('MFCC')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}_analysis.png")
    plt.close()
    
    print(f"Saved visualization for {filename} (digit {digit})")

def main():
    """Main function for debugging audio data"""
    print("Loading dataset...")
    try:
        (Xtr, ytr), (Xv, yv), (Xte, yte) = load_fsdd()
        
        # Create artifacts directory
        ensure_dir("artifacts/debug")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Print distribution of digits
    print("\nClass distribution:")
    train_counts = Counter(ytr)
    val_counts = Counter(yv)
    test_counts = Counter(yte)
    
    for i in range(10):
        print(f"  Digit {i}: {train_counts[i]} train, {val_counts[i]} val, {test_counts[i]} test")
    
    # Visualize a few samples from each class
    print("\nVisualizing samples from each class...")
    
    for digit in range(10):
        # Get samples for this digit
        samples = [path for path, label in zip(Xtr, ytr) if label == digit][:3]
        
        for sample in samples:
            visualize_audio(sample)
    
    print("\nVisualization complete! Check the artifacts/debug directory.")

if __name__ == "__main__":
    main()
