# Multi-Modal-Deepfake-Detection-Assistant-2
import cv2
import librosa
import numpy as np
import matplotlib.pyplot as plt

def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_frames)

    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        if len(frames) == num_frames:
            break
    cap.release()
    return np.array(frames)

def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mels = librosa.power_to_db(mels, ref=np.max)
    log_mels = log_mels[:128, :128]  # Resize to fixed size
    return np.expand_dims(log_mels, axis=-1)

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

video_model = load_model("models/video_model.h5")

def predict_video(frames):
    processed = preprocess_input(frames.astype(np.float32))
    preds = video_model.predict(processed)
    return np.mean(preds)  # average confidence across frames

import numpy as np
from tensorflow.keras.models import load_model

audio_model = load_model("models/audio_model.h5")

def predict_audio(spectrogram):
    input_data = np.expand_dims(spectrogram, axis=0)
    prediction = audio_model.predict(input_data)
    return prediction[0][0]

import streamlit as st
from utils import extract_frames, audio_to_spectrogram
from video_model import predict_video
from audio_model import predict_audio

st.set_page_config(page_title="DeepSecure - Deepfake Detector", layout="centered")

st.title("🛡️ DeepSecure – Multi-Modal Deepfake Detection Assistant")
st.markdown("Equip individuals, journalists, and platforms to verify the **authenticity of media** with the power of AI.")

file_type = st.radio("Choose file type to analyze:", ["Video", "Audio"])

if file_type == "Video":
    video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if video:
        with open("temp_video.mp4", "wb") as f:
            f.write(video.read())
        st.video("temp_video.mp4")

        if st.button("Analyze Video"):
            st.info("Extracting frames and analyzing...")
            frames = extract_frames("temp_video.mp4")
            score = predict_video(frames)
            verdict = "Fake" if score > 0.5 else "Real"
            st.success(f"🎬 Video is **{verdict}** (Confidence: {score:.2%})")

elif file_type == "Audio":
    audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if audio:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio.read())
        st.audio("temp_audio.wav")

        if st.button("Analyze Audio"):
            st.info("Converting to spectrogram and analyzing...")
            spec = audio_to_spectrogram("temp_audio.wav")
            score = predict_audio(spec)
            verdict = "Fake" if score > 0.5 else "Real"
            st.success(f"🎧 Audio is **{verdict}** (Confidence: {score:.2%})")

