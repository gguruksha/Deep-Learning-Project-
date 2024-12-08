import streamlit as st
import os
import torch
from inference_and_annotate import annotate_video  # Import the annotate_video function
from video_resnet import ViolenceClassifier  # Import the ViolenceClassifier
import shutil

# Streamlit UI setup
st.set_page_config(page_title="Violence Detection", page_icon=":movie_camera:")

# Set constants
TEMP_VIDEO_PATH = "temp_video.mp4"
ANNOTATED_VIDEO_PATH = "annotated_video.mp4"
MODEL_PATH = "model_resnet.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model():
    model = ViolenceClassifier(num_classes=1)
    state_dict = torch.load("model_resnet.pt", map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(DEVICE)
    return model


model = load_model()

# Streamlit UI
st.title("Violent vs Non-Violent Video Classification with Annotation")

uploaded_video = st.file_uploader("Upload a video (mp4 format)", type=["mp4"])

if uploaded_video is not None:
    with open(TEMP_VIDEO_PATH, "wb") as f:
        f.write(uploaded_video.read())

    st.write("Processing the uploaded video...")

    # Perform annotation and classification
    try:
        annotate_video(
            video_path=TEMP_VIDEO_PATH,
            output_path=ANNOTATED_VIDEO_PATH,
            model=model,
            device=DEVICE,
            clip_len=16,
            overlap=8,
            threshold=0.5,
        )
        st.success("The video has been processed and annotated successfully!")

        # Display annotated video
        st.video(ANNOTATED_VIDEO_PATH)

    except Exception as e:
        st.error(f"Error during video processing: {e}")

    # Cleanup temporary files
    if os.path.exists(TEMP_VIDEO_PATH):
        os.remove(TEMP_VIDEO_PATH)


#### CHANGES ###
#
# model = ViolenceClassifierInference()
#
# # Streamlit UI setup
# st.set_page_config(page_title="Violence Detection", page_icon=":movie_camera:")
#
# # Set constants
# TEMP_VIDEO_PATH = "temp_video.mp4"
#
# # Streamlit UI
# st.title("Violent vs Non-Violent Video Classification")
#
# uploaded_video = st.file_uploader("Upload a video (mp4 format)", type=["mp4"])
#
# def classify_video(path):
#     clips = get_clips(path)
#     output = model.infer(clips)
#     if np.max(output) > 0.5:
#         return "Violent"
#     else:
#         return "Non-Violent"
#
# if uploaded_video is not None:
#     # Save uploaded video temporarily
#     with open(TEMP_VIDEO_PATH, "wb") as f:
#         f.write(uploaded_video.read())
#
#     st.write("Processing the uploaded video...")
#
#     # Perform classification
#     try:
#         result = classify_video(TEMP_VIDEO_PATH)
#         st.success(f"The video is classified as: **{result}**")
#     except Exception as e:
#         st.error(f"Error during video classification: {e}")
#
#     # Cleanup temporary video
#     if os.path.exists(TEMP_VIDEO_PATH):
#         os.remove(TEMP_VIDEO_PATH)
