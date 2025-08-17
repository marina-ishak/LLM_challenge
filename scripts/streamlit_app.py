import streamlit as st
import numpy as np
import tensorflow as tf
from pathlib import Path
from io import BytesIO
import soundfile as sf

st.set_page_config(page_title="Digit Audio Classifier", page_icon="🎙️")

st.title("🎙️ Digit Audio Classifier (FSDD)")
st.caption("Upload a WAV or record audio (if supported), and get a prediction.")

model_path = st.text_input("Model path", value="artifacts/model.keras")
features = st.selectbox("Features", ["mfcc","logmel"])
with_deltas = st.checkbox("Add deltas", False)

uploaded = st.file_uploader("Upload WAV", type=["wav"])

def featurize_bytes(b, features="mfcc", with_deltas=False):
    from src.train import featurize
    bio = BytesIO(b)
    # need to save to temp to reuse pipeline
    tmp = Path("artifacts/_tmp.wav")
    tmp.parent.mkdir(exist_ok=True, parents=True)
    tmp.write_bytes(b)
    F = featurize(str(tmp), features, with_deltas, max_frames=80)
    return F[np.newaxis, ..., np.newaxis]

if st.button("Predict") and (uploaded is not None):
    model = tf.keras.models.load_model(model_path, compile=False)
    arr = uploaded.read()
    X = featurize_bytes(arr, features, with_deltas)
    probs = model.predict(X, verbose=0)[0]
    pred = int(np.argmax(probs))
    st.success(f"Prediction: **{pred}**")
    st.write("Probabilities:", np.round(probs, 3))
