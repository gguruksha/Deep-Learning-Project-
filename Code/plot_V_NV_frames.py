#%%
import os
import cv2
import random
import matplotlib.pyplot as plt
print(os.getcwd())
"""
Notes:
Save the Data under Data folder of root of root dir
Takes random 5 videos from V and NV dir
Extracts middle 5 frames
Plot them in 10X5 grid
"""
#%%
data_dir = os.path.abspath("../Data/Real Life Violence Dataset") # Taking the directory from the root of root dir
violence_dir = os.path.join(data_dir , 'Violence')
non_violence_dir = os.path.join(data_dir , 'NonViolence')
print(f"Data DIR: {data_dir}")
print(f"Violence DIR: {violence_dir}")
print(f"Non Violence DIR: {non_violence_dir}")


#%%
def get_random_videos(directory, num_videos=5):
    video_files = [file for file in os.listdir(directory) if file.endswith(('.mp4'))]
    selected_videos = random.sample(video_files, min(num_videos, len(video_files)))
    return selected_videos

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
    fig.savefig("frames_grid.png", bbox_inches='tight')
    plt.show()

plot_frames(v_dir, nv_dir)
