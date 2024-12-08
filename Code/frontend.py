
### CHANGES

import streamlit as st
import os
import torch
from torchvision.transforms import Compose, Resize, ToTensor
from Frame_Extraction import extract_frames  # Corrected import
from video_resnet import ViolenceClassifier
import shutil
import cv2
from PIL import Image

# Streamlit UI setup
st.set_page_config(page_title="Violence Detection", page_icon=":movie_camera:")

# Set constants
TEMP_VIDEO_PATH = "temp_video.mp4"
TEMP_FRAMES_DIR = "temp_frames"
MODEL_PATH = "model_resnet.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure temp directories exist
os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)

# Streamlit UI
st.title("Violent vs Non-Violent Video Classification")

uploaded_video = st.file_uploader("Upload a video (mp4 format)", type=["mp4"])

if uploaded_video is not None:
    # Save uploaded video temporarily
    with open(TEMP_VIDEO_PATH, "wb") as f:
        f.write(uploaded_video.read())

    st.write("Processing the uploaded video...")

    # Perform classification
    try:
        result = classify_video(TEMP_VIDEO_PATH, model, DEVICE)
        st.success(f"The video is classified as: **{result}**")
    except Exception as e:
        st.error(f"Error during video classification: {e}")

    # Cleanup temporary video
    if os.path.exists(TEMP_VIDEO_PATH):
        os.remove(TEMP_VIDEO_PATH)
