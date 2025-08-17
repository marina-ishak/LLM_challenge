import argparse, sys, numpy as np
import tensorflow as tf
from .train import featurize
from .data import read_wav
try:
    import sounddevice as sd
except Exception:
    sd = None

DIGIT_NAMES = [str(i) for i in range(10)]

def predict_file(model, wav_path, features, with_deltas):
    F = featurize(wav_path, features, with_deltas, max_frames=80)[np.newaxis, ..., np.newaxis]
    probs = model.predict(F, verbose=0)[0]
    pred = int(np.argmax(probs))
    return pred, probs

def record_and_predict(model, seconds=1.0, sr=8000, features="mfcc", with_deltas=False):
    if sd is None:
        raise RuntimeError("sounddevice not available. Use --wav instead of --mic.")
    print(f"Recording {seconds}s...")
    x = sd.rec(int(seconds*sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    x = x[:,0]
    import soundfile as sf
    sf.write("artifacts/last_mic.wav", x, sr)
    # save then run via featurize path
    pred, probs = predict_file(model, "artifacts/last_mic.wav", features, with_deltas)
    return pred, probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", type=str, help="Path to WAV file")
    ap.add_argument("--mic", action="store_true", help="Use microphone recording (1s)")
    ap.add_argument("--features", choices=["mfcc","logmel"], default="mfcc")
    ap.add_argument("--with-deltas", action="store_true")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)

    if args.mic:
        pred, probs = record_and_predict(model, features=args.features, with_deltas=args.with_deltas)
    elif args.wav:
        pred, probs = predict_file(model, args.wav, args.features, args.with_deltas)
    else:
        print("Please supply --wav <file.wav> or --mic")
        sys.exit(1)

    print("Prediction:", DIGIT_NAMES[pred])
    print("Probs:", np.round(probs, 3))

if __name__ == "__main__":
    main()
