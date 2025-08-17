"""
Interactive Spoken Digit Recognition Demo (v2-compatible)
- Matches training pipeline: 8 kHz, MAX_FRAMES=100, saved scaler
- Optional ensemble prediction (noise + slight shift)
"""

import os, sys, time, argparse
import numpy as np
import tensorflow as tf
import sounddevice as sd
import soundfile as sf

# Allow "python scripts/live_demo.py" to find src.*
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import read_wav  # optional if you ever want to test from file
from src.features import pre_emphasis, vad_trim, extract_mfcc, extract_logmel, add_deltas, pad_or_crop, normalize_audio
from src.train import featurize  # Import the exact same featurize function used in training

SAMPLE_RATE = 8000          # MUST match training
RECORD_SECONDS = 1.2        # a bit >1s helps VAD/centering
ENERGY_THRESHOLD = 1e-5     # lower threshold for more sensitivity

class LiveDigitRecognizer:
    def __init__(self, model_path, features="mfcc", with_deltas=False, sample_rate=SAMPLE_RATE):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.features = features
        self.with_deltas = with_deltas
        self.sr = sample_rate
        
        # We'll use per-batch normalization instead of a saved scaler
        self.use_dynamic_normalization = True
        
        # Warm up for consistent timing
        dummy_feat_dim = 20 if features == "mfcc" else 40
        max_frames = 80  # This should match what was used in training
        dummy = np.zeros((1, dummy_feat_dim * (3 if with_deltas else 1), max_frames, 1), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)

    def _featurize_and_norm(self, x):
        """Waveform -> feature -> normalize -> batch+channel dims"""
        # Use the EXACT same featurize function from training
        try:
            # Save audio to a temporary file
            temp_file = os.path.join("artifacts", "temp_audio.wav")
            sf.write(temp_file, x, self.sr)
            
            # Use the same featurization function as in training
            F = featurize(temp_file, features=self.features, with_deltas=self.with_deltas, max_frames=80)
            
            # Normalize features
            if self.use_dynamic_normalization:
                F = (F - np.mean(F)) / (np.std(F) + 1e-6)
            
            # Add batch and channel dimensions
            return F[np.newaxis, ..., np.newaxis]  # (1, feat, time, 1)
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Fallback to direct processing if the above fails
            x = normalize_audio(pre_emphasis(vad_trim(x, self.sr)))
            
            if self.features == "mfcc":
                F = extract_mfcc(x, self.sr, n_mfcc=20, n_mels=40)
            else:
                F = extract_logmel(x, self.sr, n_mels=40)
            
            if self.with_deltas:
                F = add_deltas(F)
                
            F = pad_or_crop(F, max_frames=80)
            
            if self.use_dynamic_normalization:
                F = (F - np.mean(F)) / (np.std(F) + 1e-6)
            
            return F[np.newaxis, ..., np.newaxis]  # (1, feat, time, 1)

    def predict_digit(self, audio_array, use_ensemble=True, debug=False):
        """
        Returns (pred_digit, confidence, inference_ms, probs[10])
        """
        preds = []

        # Original
        X = self._featurize_and_norm(audio_array)
        if debug:
            print(f"Feature shape: {X.shape}, Range: [{X.min():.4f}, {X.max():.4f}], Mean: {X.mean():.4f}, Std: {X.std():.4f}")
        
        t0 = time.time()
        p = self.model.predict(X, verbose=0)[0]
        t1 = time.time()
        preds.append(p)
        
        if debug:
            print(f"Base prediction: {np.argmax(p)} with confidence {np.max(p):.4f}")
            print(f"All probabilities: {p}")

        if use_ensemble:
            # Slight noise
            x_noisy = audio_array + 0.003 * np.random.randn(len(audio_array)).astype(np.float32)
            Xn = self._featurize_and_norm(x_noisy)
            noise_pred = self.model.predict(Xn, verbose=0)[0]
            preds.append(noise_pred)
            if debug:
                print(f"Noise prediction: {np.argmax(noise_pred)} with confidence {np.max(noise_pred):.4f}")

            # Small circular shift (~10 ms)
            shift = int(0.01 * self.sr)
            xs = np.roll(audio_array, shift)
            Xs = self._featurize_and_norm(xs)
            shift_pred = self.model.predict(Xs, verbose=0)[0]
            preds.append(shift_pred)
            if debug:
                print(f"Shift prediction: {np.argmax(shift_pred)} with confidence {np.max(shift_pred):.4f}")

        avg = np.mean(np.stack(preds, axis=0), axis=0)
        pred = int(np.argmax(avg))
        conf = float(avg[pred])
        infer_ms = (t1 - t0) * 1000.0
        
        if debug:
            print(f"Ensemble prediction: {pred} with confidence {conf:.4f}")
            print(f"Inference time: {infer_ms:.1f}ms")
            
        return pred, conf, infer_ms, avg

    def record_once(self, seconds=RECORD_SECONDS):
        print(f"Recording {seconds:.2f}s at {self.sr} Hz…")
        x = sd.rec(int(seconds * self.sr), samplerate=self.sr, channels=1, dtype=np.float32)
        sd.wait()
        x = x[:, 0]

        # Quick energy check
        energy = float(np.mean(x ** 2))
        if energy < ENERGY_THRESHOLD:
            print(f"⚠️ Low energy ({energy:.6e}) — speak louder/closer or raise threshold.")
        # Save last audio for debugging
        os.makedirs("artifacts", exist_ok=True)
        sf.write("artifacts/last_mic.wav", x, self.sr)
        return x

def main():
    ap = argparse.ArgumentParser(description="Live Digit Recognition (v2-compatible)")
    ap.add_argument("--model", default="artifacts/model.keras")
    ap.add_argument("--features", choices=["mfcc", "logmel"], default="mfcc")
    ap.add_argument("--with-deltas", action="store_true", 
                    help="Use delta features (same as used during training)")
    ap.add_argument("--num", type=int, default=5, help="How many trials to run")
    ap.add_argument("--no-ensemble", action="store_true", help="Disable inference-time ensemble")
    ap.add_argument("--threshold", type=float, default=ENERGY_THRESHOLD, 
                    help="Energy threshold for voice detection (try 1e-5 to 1e-3)")
    ap.add_argument("--debug", action="store_true", help="Show detailed debugging info")
    args = ap.parse_args()

    # Basic audio sanity check
    try:
        devices = sd.query_devices()
        print("Available audio devices:")
        for i, dev in enumerate(devices):
            if dev.get('max_input_channels', 0) > 0:  # Input device
                print(f"  {i}: {dev['name']} (Input)")
        print("")
    except Exception as e:
        print(f"Audio device error: {e}\nMake sure microphone permissions are granted.")
        sys.exit(1)

    rec = LiveDigitRecognizer(
        model_path=args.model,
        features=args.features,
        with_deltas=args.with_deltas,
        sample_rate=SAMPLE_RATE
    )

    print("\n🎤 Say a digit (0–9) after each prompt.")
    print(f"Current energy threshold: {args.threshold:.6e}")
    print("If you see low energy warnings, try with --threshold 1e-6\n")
    
    try:
        # Initial audio check
        print("Testing microphone (speak normally)...")
        test_audio = sd.rec(int(1.0 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
        sd.wait()
        test_energy = float(np.mean(test_audio ** 2))
        print(f"Detected energy: {test_energy:.6e}")
        if test_energy < args.threshold:
            print("WARNING: Audio level very low. Consider these options:")
            print("1. Speak louder or get closer to the microphone")
            print("2. Check your microphone settings in Windows")
            print("3. Use --threshold parameter with a lower value (e.g. 1e-6)")
            if input("Continue anyway? (y/n): ").lower() != 'y':
                sys.exit(0)
        print("\nStarting recognition...\n")
    except Exception as e:
        print(f"Microphone test error: {e}")
    
    energy_threshold = args.threshold
    
    for i in range(1, args.num + 1):
        input(f"[{i}/{args.num}] Press Enter, then speak…")
        x = rec.record_once(seconds=RECORD_SECONDS)
        
        # Check audio energy
        energy = float(np.mean(x ** 2))
        if energy < energy_threshold:
            print(f"⚠️ Low energy ({energy:.6e}) — speak louder/closer or use --threshold {energy*0.8:.6e}")
        
        digit, conf, ms, probs = rec.predict_digit(x, use_ensemble=not args.no_ensemble, debug=args.debug)
        
        # Show more detailed info
        print(f"→ Predicted: {digit} | Confidence: {conf:.3f} | Inference: {ms:.1f} ms")
        
        # Show top 3 predictions
        top3 = np.argsort(probs)[-3:][::-1]
        print(f"  Top 3 predictions: {top3[0]}({probs[top3[0]]:.3f}), {top3[1]}({probs[top3[1]]:.3f}), {top3[2]}({probs[top3[2]]:.3f})")

if __name__ == "__main__":
    main()
