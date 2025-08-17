"""
Simple command-line real-time audio classification
"""
import os
import sys
import time
import argparse
import numpy as np
import pyaudio
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

class AudioProcessor:
    def __init__(self, model, feature_type="mfcc", with_deltas=False, window_size=16000):
        self.model = model
        self.feature_type = feature_type
        self.with_deltas = with_deltas
        self.window_size = window_size
        self.buffer = np.zeros(window_size, dtype=np.float32)
        self.queue = Queue()
        self.running = True
        self.last_prediction = None
        self.last_confidence = 0
    
    def callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def add_audio(self, audio_data):
        """Add audio to rolling buffer"""
        if len(audio_data) >= self.window_size:
            # If audio data is larger than buffer, just take the latest part
            self.buffer = audio_data[-self.window_size:]
        else:
            # Shift buffer and add new data
            self.buffer = np.roll(self.buffer, -len(audio_data))
            self.buffer[-len(audio_data):] = audio_data
        return self.buffer.copy()
    
    def process_audio(self):
        """Process audio data and make predictions"""
        try:
            while self.running:
                # Process audio data in chunks
                audio_data = np.array([])
                
                # Get all available audio data from the queue with a timeout
                while not self.queue.empty():
                    chunk = self.queue.get()
                    audio_data = np.concatenate((audio_data, chunk))
                
                if len(audio_data) > 0:
                    # Add to rolling buffer
                    buffer_data = self.add_audio(audio_data)
                    
                    # Extract features
                    features = self.get_features(buffer_data)
                    
                    # Perform inference
                    start_time = time.time()
                    predictions = self.model.predict(features, verbose=0)[0]
                    inference_time = (time.time() - start_time) * 1000  # in ms
                    
                    # Get prediction
                    predicted_digit = np.argmax(predictions)
                    confidence = predictions[predicted_digit]
                    
                    # Only update if confidence is high enough
                    if confidence > max(0.5, self.last_confidence * 0.8):
                        self.last_prediction = predicted_digit
                        self.last_confidence = confidence
                        
                        # Clear the console (works for both Windows and Unix)
                        os.system('cls' if os.name == 'nt' else 'clear')
                        
                        # Print prediction
                        print("=" * 40)
                        print(f"PREDICTED DIGIT: {predicted_digit}")
                        print("=" * 40)
                        print(f"Confidence: {confidence:.4f}")
                        print(f"Inference time: {inference_time:.2f} ms")
                        print("\nProbabilities:")
                        for i, prob in enumerate(predictions):
                            bar_length = int(50 * prob)
                            bar = '█' * bar_length + '░' * (50 - bar_length)
                            print(f"Digit {i}: {prob:.4f} |{bar}|")
                        
                        print("\nPress Ctrl+C to exit")
                
                # Sleep a bit to reduce CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.running = False
            print("\nStopping audio processing...")
    
    def get_features(self, audio_data):
        """Extract features from audio data"""
        # Process audio
        audio_data = normalize_audio(pre_emphasis(vad_trim(audio_data, RATE)))
        
        # Extract features
        if self.feature_type == "mfcc":
            features = extract_mfcc(audio_data, RATE, n_mfcc=20, n_mels=40)
        else:
            features = extract_logmel(audio_data, RATE, n_mels=40)
        
        # Add deltas if requested
        if self.with_deltas:
            features = add_deltas(features)
        
        # Pad or crop
        features = pad_or_crop(features, max_frames=80)
        
        # Add batch and channel dimensions
        features = np.expand_dims(np.expand_dims(features, 0), -1)
        
        return features
    
    def start(self):
        """Start audio processing"""
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Start audio stream
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self.callback
        )
        
        # Start processing thread
        self.thread = Thread(target=self.process_audio)
        self.thread.daemon = True
        self.thread.start()
        
        print("Starting real-time audio classification...")
        print("Speak a digit (0-9) into your microphone.")
        print("Press Ctrl+C to exit.")
        
        try:
            self.stream.start_stream()
            # Keep main thread alive
            while self.running and self.thread.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop audio processing"""
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
        print("Audio processing stopped.")

def main():
    parser = argparse.ArgumentParser(description="Simple real-time audio classification")
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
    
    # Create audio processor
    processor = AudioProcessor(
        model=model,
        feature_type=args.features,
        with_deltas=args.with_deltas
    )
    
    # Start processing
    processor.start()

if __name__ == "__main__":
    main()
