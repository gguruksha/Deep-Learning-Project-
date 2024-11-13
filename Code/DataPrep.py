import os
import cv2
import pandas as pd

# Define paths
ROOT_DIR = os.getcwd()
os.chdir("..")
DATA_DIR = os.getcwd() + '/Data'
dataset_path = DATA_DIR + '/Real Life Violence Dataset/'
violence_path = os.path.join(dataset_path, 'Violence')
non_violence_path = os.path.join(dataset_path, 'NonViolence')

# Initialize lists for data and labels
data = []
labels = []

# Define a function to load videos and assign labels
def load_videos_from_directory(directory, label):
    for filename in os.listdir(directory):
        if filename.endswith('.mp4'):
            filepath = os.path.join(directory, filename)
            data.append(filepath)
            labels.append(label)

# Load videos from both classes
load_videos_from_directory(violence_path, 1)        # Label 1 for violence
load_videos_from_directory(non_violence_path, 0)    # Label 0 for non-violence

# Create a DataFrame to manage paths and labels
dataset_df = pd.DataFrame({
    'video_path': data,
    'label': labels
})

# Save
dataset_df.to_csv('video_labels.csv', index=False)

#-----------------------------------------------------------------------

# Define output directory for frames
output_dir = DATA_DIR + '/Preprocessed_Frames'
os.makedirs(output_dir, exist_ok=True)
violence_output_dir = output_dir+'/Violence'
os.makedirs(output_dir+'/Violence', exist_ok=True)
nonviolence_output_dir = output_dir+'/NonViolence'
os.makedirs(output_dir+'/NonViolence', exist_ok=True)


# Frame extraction function
def extract_frames(video_path, output_folder, label, frame_rate=1, img_size=(224, 224)):
    """
    Extract frames from a video and save them as images.
    :param video_path: Path to the video file.
    :param output_folder: Folder to save extracted frames.
    :param label: Label for the video.
    :param frame_rate: Number of frames to capture per second.
    :param img_size: Size to resize frames to (width, height).
    """
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Capture frame every `frame_rate` seconds
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % int(cap.get(cv2.CAP_PROP_FPS) * frame_rate) == 0:
            # Resize frame
            frame = cv2.resize(frame, img_size)
            # Save frame as image file
            frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count}_label_{label}.jpg")
            print(frame_filename)
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

    cap.release()


# Process each video and save frames
for idx, row in dataset_df.iterrows():
    # print("\n")
    print(idx)
    label = row['label']
    video_path = row['video_path']
    if label == 1:
        output_dir = violence_output_dir
    else:
        output_dir = nonviolence_output_dir
    extract_frames(video_path, output_dir, label)



