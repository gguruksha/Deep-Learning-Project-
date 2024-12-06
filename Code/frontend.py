import streamlit as st

st.set_page_config(page_title="Video Chat Bot", page_icon=":movie_camera:")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Video Chat Bot")

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

uploaded_video = st.file_uploader("Upload a video (Format: mp4)", type=["mp4"])

if uploaded_video is not None:
    user_message = f"**User uploaded video:** `{uploaded_video.name}`"
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)

    assistant_response = f"Thanks for uploading the video: `{uploaded_video.name}`! Let me see what I can process..."
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.write(assistant_response)
