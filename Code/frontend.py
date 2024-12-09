
import streamlit as st
import os
import torch
from inference_and_annotate import annotate_video
from video_resnet import ViolenceClassifier
import cv2

# Streamlit UI setup
st.set_page_config(page_title="Violence Detection", page_icon=":movie_camera:")

# Constants
TEMP_VIDEO_PATH = "temp_video.mp4"
ANNOTATED_VIDEO_PATH = "annotated_video.mp4"
MODEL_PATH = "model_resnet.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    /* Apply font to the entire app */
    body, .title, .subtitle, .instructions, .metadata, .footer {
        font-family: 'Roboto', sans-serif;
    }

    /* Style for the main title */
    .title {
        font-size: 36px;
        color: #FF6347;
        text-align: center;
        margin-bottom: -10px;
    }

    /* Style for the subtitle */
    .subtitle {
        font-size: 24px;
        color: #333333;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Style for instructions */
    .instructions {
        font-size: 18px;
        color: #555555;
        margin-bottom: 20px;
    }

    /* Style for metadata */
    .metadata {
        font-size: 16px;
        color: #333333;
        margin-bottom: 10px;
    }

    /* Style for the download button */
    .download-button {
        text-align: center;
        margin-top: 20px;
    }

    /* Style for the footer */
    .footer {
        font-size: 14px;
        color: #888888;
        text-align: center;
        margin-top: 40px;
    }

    /* Add a subtle background color */
    .reportview-container {
        background-color: #F5F5F5;
    }

    /* Style for the processing spinner */
    .spinner {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Load model
@st.cache_resource
def load_model():
    model = ViolenceClassifier(num_classes=1)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(DEVICE)
    return model


model = load_model()

# UI Elements
st.markdown('<div class="title">Violence Detection in Videos</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">Classify and Annotate Videos as Violent or Non-Violent</div>', unsafe_allow_html=True)

st.markdown(
    """
    ### Steps:
    1. Upload a video in **MP4 format**.
    2. The video will be processed, and violent or non-violent segments will be annotated.
    3. Viola! Download the annotated video after processing.
    """
)

st.sidebar.header("Model Details")
st.sidebar.write("**Model:** ResNet-3D (r3d_18)")
st.sidebar.write("**Clip Length:** 16 frames")
st.sidebar.write("**Overlap:** 8 frames")
st.sidebar.write("**Threshold:** 0.5 for Violent classification")
st.sidebar.write("**Device:** {}".format("GPU" if torch.cuda.is_available() else "CPU"))

# Video Upload
uploaded_video = st.file_uploader("📥 **Upload a video (MP4 format)**", type=["mp4"])

if uploaded_video is not None:
    with open(TEMP_VIDEO_PATH, "wb") as f:
        f.write(uploaded_video.read())

    # Display video metadata
    cap = cv2.VideoCapture(TEMP_VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    st.write(f"**Video Duration:** {duration:.2f} seconds")
    st.write(f"**Resolution:** {resolution[0]} x {resolution[1]}")
    st.write(f"**Frames Per Second (FPS):** {fps:.2f}")
    if duration > 120:
        st.warning("⚠️ Large video detected. Processing may take longer.")

    st.write("Processing the uploaded video...")

    # Processing
    try:
        with st.spinner("Analyzing and annotating the video..."):
            annotate_video(
                video_path=TEMP_VIDEO_PATH,
                output_path=ANNOTATED_VIDEO_PATH,
                model=model,
                device=DEVICE,
                clip_len=16,
                overlap=8,
                threshold=0.5,
            )
        st.success("✅The video has been processed and annotated successfully!")

        # Display annotated video
        st.video(ANNOTATED_VIDEO_PATH)

        # Download button
        st.download_button(
            label="Download Annotated Video",
            data=open(ANNOTATED_VIDEO_PATH, "rb"),
            file_name="annotated_video.mp4",
            mime="video/mp4",
        )

    except Exception as e:
        st.error(f"❌Error during video processing: {e}")

    # Cleanup temporary files
    if os.path.exists(TEMP_VIDEO_PATH):
        os.remove(TEMP_VIDEO_PATH)

# EARLIER CHANGES####
# import streamlit as st
# import os
# import torch
# from inference_and_annotate import annotate_video  # Import the annotate_video function
# from video_resnet import ViolenceClassifier  # Import the ViolenceClassifier
# import shutil
#
# # Streamlit UI setup
# st.set_page_config(page_title="Violence Detection", page_icon=":movie_camera:")
#
# # Set constants
# TEMP_VIDEO_PATH = "temp_video.mp4"
# ANNOTATED_VIDEO_PATH = "annotated_video.mp4"
# MODEL_PATH = "model_resnet.pt"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Load model
# @st.cache_resource
# def load_model():
#     model = ViolenceClassifier(num_classes=1)
#     state_dict = torch.load("model_resnet.pt", map_location=DEVICE)
#     model.load_state_dict(state_dict, strict=True)
#     model.eval()
#     model.to(DEVICE)
#     return model
#
#
# model = load_model()
#
# # Streamlit UI
# st.title("Violent vs Non-Violent Video Classification with Annotation")
#
# uploaded_video = st.file_uploader("Upload a video (mp4 format)", type=["mp4"])
#
# if uploaded_video is not None:
#     with open(TEMP_VIDEO_PATH, "wb") as f:
#         f.write(uploaded_video.read())
#
#     st.write("Processing the uploaded video...")
#
#     # Perform annotation and classification
#     try:
#         annotate_video(
#             video_path=TEMP_VIDEO_PATH,
#             output_path=ANNOTATED_VIDEO_PATH,
#             model=model,
#             device=DEVICE,
#             clip_len=16,
#             overlap=8,
#             threshold=0.5,
#         )
#         st.success("The video has been processed and annotated successfully!")
#
#         # Display annotated video
#         st.video(ANNOTATED_VIDEO_PATH)
#
#     except Exception as e:
#         st.error(f"Error during video processing: {e}")
#
#     # Cleanup temporary files
#     if os.path.exists(TEMP_VIDEO_PATH):
#         os.remove(TEMP_VIDEO_PATH)