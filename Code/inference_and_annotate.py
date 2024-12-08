import os
import torch
import cv2
from torchvision.io import read_video
from torch.nn.functional import sigmoid
from dataloader import get_data_loader
from video_resnet import ViolenceClassifier
from torchvision.models.video import r3d_18, R3D_18_Weights
import numpy as np

def annotate_video(video_path, output_path, model, device, clip_len=16, overlap=8, threshold=0.5):
    """
    Annotates a single video by detecting violence and saving the annotated video.
    """
    video, _, _ = read_video(video_path, output_format="TCHW", pts_unit="sec")
    video = video.float() / 255.0  # Normalize to range [0, 1]

    num_frames, height, width, channels = video.shape[0], video.shape[2], video.shape[3], video.shape[1]
    fps = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))

    # VideoWriter to stitch clips back together
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model.eval()
    with torch.no_grad():
        for start_idx in range(0, num_frames - clip_len + 1, clip_len - overlap):
            # Extract clip
            clip = video[start_idx:start_idx + clip_len]
            clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, C, T, H, W]

            # Predictions
            prediction = sigmoid(model(clip)).item()
            is_violent = prediction >= threshold
            print("Sigmoid Prediction: {} is_violent: {}".format(prediction, is_violent))

            # Annotate frames in the clip
            for frame_idx in range(clip_len):
                if start_idx + frame_idx >= num_frames:
                    break
                frame_tensor = video[start_idx + frame_idx].permute(1, 2, 0).cpu().numpy()  # TCHW -> HWC
                frame_bgr = cv2.cvtColor(frame_tensor, cv2.COLOR_RGB2BGR)

                # Scale back to 0-255 and convert to uint8
                frame_bgr = (frame_bgr * 255).astype(np.uint8)

                label = "Violence" if is_violent else "No Violence"
                color = (0, 0, 255) if is_violent else (0, 255, 0)
                cv2.putText(frame_bgr, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.rectangle(frame_bgr, (10, 10), (width - 10, height - 10), color, 3)
                out.write(frame_bgr)

    out.release()
    print(f"Annotated video saved to {output_path}")

def process_test_dataloader(test_dataloader, output_folder, model, device, clip_len=16, overlap=8, threshold=0.53):
    """
    Processes all videos in the test dataloader and saves annotated versions in the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    for batch_idx, (clips, labels) in enumerate(test_dataloader):
        print(batch_idx)
        for idx in range(clips.shape[0]):
            video_path = test_dataloader.dataset.files_labels['path'][batch_idx * test_dataloader.batch_size + idx]
            print(video_path)
            video_name = os.path.basename(video_path)
            output_path = os.path.join(output_folder, f"annotated_{video_name}")
            annotate_video(video_path, output_path, model, device, clip_len, overlap, threshold)

if __name__ == '__main__':
    # Load model and test dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViolenceClassifier(num_classes=1)
    model.load_state_dict(torch.load("model_resnet.pt"))
    model.to(device)

    train_dl, test_dl = get_data_loader(R3D_18_Weights.DEFAULT.transforms(), clip_len=16)

    output_folder = "./annotated_videos"
    process_test_dataloader(test_dl, output_folder, model, device)
