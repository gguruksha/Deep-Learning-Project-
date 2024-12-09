import streamlit as st
import os
import torch
import cv2
from run_inference import ViolenceClassifierInference, get_clips, stitch_clips_with_annotations
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read("config.conf")

# Streamlit UI setup
st.set_page_config(page_title="Violence Detection", page_icon=":movie_camera:")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    body, .title, .subtitle, .instructions, .metadata, .footer {
        font-family: 'Roboto', sans-serif;
    }
    .title {
        font-size: 36px;
        color: #FF6347;
        text-align: center;
        margin-bottom: -10px;
    }
    .subtitle {
        font-size: 24px;
        color: #333333;
        text-align: center;
        margin-bottom: 20px;
    }
    .instructions {
        font-size: 18px;
        color: #555555;
        margin-bottom: 20px;
    }
    .metadata {
        font-size: 16px;
        color: #FFFFFF;
        margin-bottom: 10px;
    }
    .download-button {
        text-align: center;
        margin-top: 20px;
    }
    .footer {
        font-size: 14px;
        color: #888888;
        text-align: center;
        margin-top: 40px;
    }
    .reportview-container {
        background-color: #F5F5F5;
    }
    .spinner {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# UI Elements
st.title("Violent vs Non-Violent Video Classification with Annotation")
st.markdown(
    """
    <div style="background-color: #000000; padding: 10px; border-radius: 10px;">
        <h3 style="color: white;">Instructions:</h3>
        <ol style="color: white;">
            <li>The video will be processed, and violent or non-violent segments will be annotated.</li>
            <li>Download the annotated video after processing.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for Model Details
st.sidebar.header("Model Details")
st.sidebar.write("Model: ResNet-3D (r3d_18)")
st.sidebar.write("Clip Length: 16 frames")
st.sidebar.write("Overlap: 8 frames")
st.sidebar.write("Threshold: 0.5 for Violent classification")
st.sidebar.write(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Set constants
TEMP_VIDEO_PATH = os.path.join(config['Streamlit']['temp_folder'], 'temp_video.mp4')
ANNOTATED_VIDEO_PATH = os.path.join(config['Streamlit']['temp_folder'], 'annotated_video.mp4')
MODEL_PATH = config['Streamlit']['model_path']
DEVICE = config['Streamlit']['device']

inference_engine = ViolenceClassifierInference(MODEL_PATH, DEVICE)

@st.cache_data(show_spinner=False)
def process_video(temp_video_path, annotated_video_path):
    clips = get_clips(temp_video_path)
    output = inference_engine.infer(clips)
    stitch_clips_with_annotations(clips, output, annotated_video_path, fps=30)
    with open(annotated_video_path, "rb") as f:
        return f.read()

uploaded_video = st.file_uploader("Upload a video (MP4 format)", type=["mp4"])

if uploaded_video:
    with open(TEMP_VIDEO_PATH, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(TEMP_VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    st.markdown(f'<div class="metadata">Video Duration: {duration:.2f} seconds</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metadata">Resolution: {resolution[0]} x {resolution[1]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metadata">Frames Per Second (FPS): {fps:.2f}</div>', unsafe_allow_html=True)

    if duration > 120:
        st.warning("⚠️ *Large video detected. Processing may take longer.*")

    try:
        with st.spinner("Analyzing and annotating the video..."):
            # clips = get_clips(TEMP_VIDEO_PATH)
            # output = inference_engine.infer(clips)
            # output_folder = config["Inference"]["output_dir"]
            # stitch_clips_with_annotations(clips, output, ANNOTATED_VIDEO_PATH, fps=30)
            video_bytes = process_video(TEMP_VIDEO_PATH, ANNOTATED_VIDEO_PATH)

        st.success("✅ The video has been processed and annotated successfully!")
        st.video(ANNOTATED_VIDEO_PATH)

    except Exception as e:
        st.error(f"❌ Error during video processing: {e}")

    # Read the annotated video into memory
    with open(ANNOTATED_VIDEO_PATH, "rb") as video_file:
        video_bytes = video_file.read()

    # Download button
    st.download_button(
        label="⬇️ *Download Annotated Video*",
        data=video_bytes,
        file_name="annotated_video.mp4",
        mime="video/mp4",
    )