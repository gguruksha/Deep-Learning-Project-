import streamlit as st
import os
import torch
import shutil
from run_inference import get_clips, ViolenceClassifierInference
import numpy as np


model = ViolenceClassifierInference()

# Streamlit UI setup
st.set_page_config(page_title="Violence Detection", page_icon=":movie_camera:")

# Set constants
TEMP_VIDEO_PATH = "temp_video.mp4"

# Streamlit UI
st.title("Violent vs Non-Violent Video Classification")

uploaded_video = st.file_uploader("Upload a video (mp4 format)", type=["mp4"])

def classify_video(path):
    clips = get_clips(path)
    output = model.infer(clips)
    if np.max(output) > 0.5:
        return "Violent"
    else:
        return "Non-Violent"

if uploaded_video is not None:
    # Save uploaded video temporarily
    with open(TEMP_VIDEO_PATH, "wb") as f:
        f.write(uploaded_video.read())

    st.write("Processing the uploaded video...")

    # Perform classification
    try:
        result = classify_video(TEMP_VIDEO_PATH)
        st.success(f"The video is classified as: **{result}**")
    except Exception as e:
        st.error(f"Error during video classification: {e}")

    # Cleanup temporary video
    if os.path.exists(TEMP_VIDEO_PATH):
        os.remove(TEMP_VIDEO_PATH)
