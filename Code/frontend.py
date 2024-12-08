
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

# Load model
@st.cache_resource
def load_model():
    model = ViolenceClassifier(num_classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

model = load_model()

# Preprocessing function for frames
def preprocess_frame(frame, img_size=(224, 224)):
    transform = Compose([
        Resize(img_size),
        ToTensor()
    ])
    # Ensure the frame is RGB
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    # Convert frame to a PIL Image for transform
    img = Image.fromarray(frame)
    tensor = transform(img)  # This gives [3, H, W]

    # Since model likely expects [N, C, D, H, W], we add two dimensions:
    # unsqueeze(0) for batch, unsqueeze(2) for depth
    tensor = tensor.unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]

    return tensor

def classify_video(video_path, model, device, frame_rate=1):
    # Extract frames as before
    extract_frames(video_path, TEMP_FRAMES_DIR, label=None, frame_rate=frame_rate)

    predictions = []
    for frame_file in os.listdir(TEMP_FRAMES_DIR):
        frame_path = os.path.join(TEMP_FRAMES_DIR, frame_file)
        frame = cv2.imread(frame_path)

        # Debug: Print frame shape
        print(f"Frame shape: {frame.shape}")

        input_tensor = preprocess_frame(frame)
        input_tensor = input_tensor.to(device)

        # Model inference
        with torch.no_grad():
            output = model(input_tensor)
            predictions.append(output.item())

    # Cleanup
    shutil.rmtree(TEMP_FRAMES_DIR)

    # Aggregate predictions
    avg_score = sum(predictions) / len(predictions)
    return "Violent" if avg_score >= 0.5 else "Non-Violent"

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
