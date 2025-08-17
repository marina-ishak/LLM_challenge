"""
Real-time audio classification from microphone input
"""
import os
import sys
import time
import argparse
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from datetime import datetime
from queue import Queue
from threading import Thread

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import pre_emphasis, vad_trim, extract_mfcc, extract_logmel, add_deltas, pad_or_crop, normalize_audio
from src.data import read_wav

# Constants
RATE = 16000  # Sample rate
CHUNK = 1024  # Buffer size
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RECORD_SECONDS = 1  # Length of audio to process at once

# Create a buffer to store audio data
q = Queue()

def callback(in_data, frame_count, time_info, status):
    """Callback function for PyAudio"""
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    q.put(audio_data)
    return (in_data, pyaudio.paContinue)

def get_features(audio_data, sr, feature_type="mfcc", with_deltas=False):
    """Extract features from raw audio data"""
    # Process audio
    audio_data = normalize_audio(pre_emphasis(vad_trim(audio_data, sr)))
    
    # Extract features
    if feature_type == "mfcc":
        features = extract_mfcc(audio_data, sr, n_mfcc=20, n_mels=40)
    else:
        features = extract_logmel(audio_data, sr, n_mels=40)
    
    # Add deltas if requested
    if with_deltas:
        features = add_deltas(features)
    
    # Pad or crop
    features = pad_or_crop(features, max_frames=80)
    
    # Add batch and channel dimensions
    features = np.expand_dims(np.expand_dims(features, 0), -1)
    
    return features

class RollingAudioBuffer:
    """Class to maintain a rolling buffer of audio data"""
    def __init__(self, window_size=RATE, hop_size=RATE//2):
        self.buffer = np.zeros(window_size, dtype=np.float32)
        self.window_size = window_size
        self.hop_size = hop_size
    
    def add_audio(self, audio_data):
        """Add new audio data to the buffer"""
        if len(audio_data) >= self.window_size:
            # If audio data is larger than buffer, just take the latest part
            self.buffer = audio_data[-self.window_size:]
        else:
            # Shift buffer and add new data
            self.buffer = np.roll(self.buffer, -len(audio_data))
            self.buffer[-len(audio_data):] = audio_data
        return self.buffer.copy()

def update_plot(frame, line, bar_container, audio_buffer, model, feature_type, with_deltas):
    """Update function for animation"""
    # Process audio data in chunks
    audio_data = np.array([])
    
    # Get all available audio data from the queue
    while not q.empty():
        chunk = q.get()
        audio_data = np.concatenate((audio_data, chunk))
    
    if len(audio_data) > 0:
        # Add to rolling buffer
        buffer_data = audio_buffer.add_audio(audio_data)
        
        # Update waveform plot
        line.set_ydata(buffer_data)
        
        # Extract features and make prediction
        features = get_features(buffer_data, RATE, feature_type, with_deltas)
        
        # Perform inference
        predictions = model.predict(features, verbose=0)[0]
        
        # Update bar chart
        for bar, height in zip(bar_container, predictions):
            bar.set_height(height)
    
    return line, bar_container

def main():
    parser = argparse.ArgumentParser(description="Real-time audio classification")
    parser.add_argument("--model", default="artifacts/model.keras", help="Path to trained model")
    parser.add_argument("--features", choices=["mfcc", "logmel"], default="mfcc", help="Feature type")
    parser.add_argument("--with-deltas", action="store_true", help="Use delta features")
    args = parser.parse_args()
    
    # Load model
    print("Loading model from", args.model)
    model = tf.keras.models.load_model(args.model)
    model.summary()
    
    # Warm up the model
    dummy_input = np.zeros((1, 60 if args.with_deltas else 20, 80, 1), dtype=np.float32)
    _ = model.predict(dummy_input)
    
    # Initialize audio
    p = pyaudio.PyAudio()
    
    # Create a rolling audio buffer
    audio_buffer = RollingAudioBuffer(window_size=RATE, hop_size=RATE//2)
    
    # Create a figure for visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)
    
    # Set up the plots
    ax1.set_title("Audio Waveform")
    ax1.set_xlim(0, RATE)
    ax1.set_ylim(-1, 1)
    line, = ax1.plot(np.arange(RATE), np.zeros(RATE))
    
    ax2.set_title("Digit Predictions")
    ax2.set_ylim(0, 1)
    ax2.set_xlim(-0.5, 9.5)
    ax2.set_xticks(range(10))
    ax2.set_xlabel("Digit")
    ax2.set_ylabel("Probability")
    bar_container = ax2.bar(range(10), np.zeros(10))
    
    # Start streaming
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=callback)
    
    # Start the stream
    stream.start_stream()
    
    print("* Recording and processing audio in real-time. Press Ctrl+C to stop.")
    
    # Create animation
    ani = FuncAnimation(
        fig, 
        update_plot, 
        fargs=(line, bar_container, audio_buffer, model, args.features, args.with_deltas),
        interval=100,  # Update every 100ms
        blit=False
    )
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Audio stream terminated.")

if __name__ == "__main__":
    main()
