#%%
import configparser
import shutil

import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
print(os.getcwd())
"""
Notes:
Save the Data under Data folder of root of root dir
Takes random 5 videos from V and NV dir
Extracts middle 5 frames
Plot them in 10X5 grid

Notes: # Updated 11/25/2024

Taking the config file to read paths to main base form.
Added frame count distplot for violence and non violence.
Updated config to add Visualization Section
Added a Visualization dir.
Creating on fly and update the Visualization dir.
"""
#%%
try:
    config = configparser.ConfigParser()
    config.read('config.conf') #Change this before pull request. Console work dir is different
    dataset_path = config.get('Dataset', 'dataset_path')
    violence_dir_path = config.get('Dataset', 'violence_directory')
    non_violence_dir_path = config.get('Dataset', 'non_violence_directory')
    data_dir = os.path.abspath(f"./{dataset_path}") # dataset dir outside code dir.
    violence_dir = os.path.join(data_dir , violence_dir_path)
    non_violence_dir = os.path.join(data_dir , non_violence_dir_path)
    print(f"Data DIR: {data_dir}")
    print(f"Violence DIR: {violence_dir}")
    print(f"Non Violence DIR: {non_violence_dir}")

    op_dir_viz = config.get('Visualizations', 'visualization_path')

    op_dir = os.path.join(os.path.abspath("..") , op_dir_viz) #Change this before pull request

    if os.path.exists(op_dir):
        shutil.rmtree(op_dir)
        os.makedirs(op_dir)
    else:
        os.makedirs(op_dir)

    print(f"OP DIR: {non_violence_dir}")

except Exception as e:
    print(e)
    print("Please update the config file.")


#%%
def get_random_videos(directory, num_videos=5):
    video_files = [file for file in os.listdir(directory) if file.endswith(('.mp4'))]
    selected_videos = random.sample(video_files, min(num_videos, len(video_files)))
    return selected_videos

#%%
def get_frame_counts(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count

#%%
def extract_middle_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_index = frame_count // 2
    start_index = max(0, middle_index - num_frames // 2)
    frame_indices = range(start_index, start_index + num_frames)
    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

#%%
V_selected_videos = get_random_videos(violence_dir, num_videos=5)
NV_selected_videos = get_random_videos(non_violence_dir , num_videos=5)
print(V_selected_videos)
print(NV_selected_videos)

#%%
violence_frame_num = []
nonviolence_frame_num = []

violence_videos = [file for file in os.listdir(violence_dir) if file.endswith(('.mp4'))]
nonviolence_videos = [file for file in os.listdir(non_violence_dir) if file.endswith(('.mp4'))]

for video in violence_videos:
    video_path = os.path.join(violence_dir, video)
    frames_count = get_frame_counts(video_path)
    violence_frame_num.append(frames_count)

for video in nonviolence_videos:
    video_path = os.path.join(non_violence_dir, video)
    frames_count = get_frame_counts(video_path)
    nonviolence_frame_num.append(frames_count)

fig = plt.figure(figsize=(10, 6))
sns.kdeplot(violence_frame_num, fill=True, alpha=0.5, label='Violence Frames Distribution', linewidth=2)
sns.kdeplot(nonviolence_frame_num, fill=True, alpha=0.5, label='Non-violence Frames Distribution', linewidth=2)
plt.xlabel('Number of Frames')
plt.ylabel('Density')
plt.title('Frame Count Distribution: Violence vs Non-violence')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

output_path = op_dir+ "/" + "violence_vs_nonviolence_distribution.png"
fig.savefig(output_path, bbox_inches='tight')

#%% For visualization
v_dir = {}
nv_dir = {}

for video in V_selected_videos:
    video_path = os.path.join(violence_dir, video)
    frames = extract_middle_frames(video_path)
    v_dir[video] = frames

for video in NV_selected_videos:
    video_path = os.path.join(non_violence_dir , video)
    frames = extract_middle_frames(video_path)
    nv_dir[video] = frames

#%%
def plot_frames(v_dir, nv_dir):
    fig, axes = plt.subplots(10, 5, figsize=(15, 30))
    axes = axes.flatten()
    all_frames = []
    for frames in v_dir.values():
        all_frames.extend(frames[:5])
    for frames in nv_dir.values():
        all_frames.extend(frames[:5])

    for i, frame in enumerate(all_frames):
        if i >= 50:
            break
        axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')

        if i < len(v_dir) * 5:
            axes[i].set_title("Violence")
        else:
            axes[i].set_title("Non-Violence")

    plt.tight_layout()
    fig.savefig(op_dir+ "/" + "frames_grid.png", bbox_inches='tight')
    plt.show()

plot_frames(v_dir, nv_dir)
