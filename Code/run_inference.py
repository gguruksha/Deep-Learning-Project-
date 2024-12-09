import torch
from video_resnet import ViolenceClassifier
from torchvision.io import read_video
from torchvision.models.video import R3D_18_Weights
import cv2
import numpy as np
import os
import configparser
import subprocess

class ViolenceClassifierInference():
    def __init__(self, model_path, device):
        self.classifier = ViolenceClassifier(num_classes=1)
        self.transforms = R3D_18_Weights.DEFAULT.transforms()
        self.classifier.load_state_dict(torch.load(model_path))
        self.device = torch.device(device)
        self.classifier.to(self.device)
        self.classifier.eval()

    def infer(self, video_frames):
        input = self.transforms(video_frames)
        input = input.to(self.device)
        with torch.no_grad():
            output = self.classifier(input)
        output = output.detach().cpu().numpy()
        return output


def get_clips(video_path):
    video, audio, metadata = read_video(video_path, output_format="TCHW", pts_unit='sec')
    clips = []
    start = 0
    end = 16
    while end < video.shape[0]:
        clips.append(video[start:end])
        start = end
        end += 16
    return torch.stack(clips, dim=0)

def stitch_clips_with_annotations(clips, output, output_path, fps=30):
    """
    Stitches clips into a single video with annotations based on the output predictions.
    Args:
        clips (torch.Tensor): Tensor of video clips, shape [num_clips, clip_len, C, H, W].
        output (numpy.ndarray): Predictions for each clip, shape [num_clips].
        output_path (str): Path to save the stitched and annotated video.
        fps (int): Frames per second for the output video.
    """
    # Extract video dimensions
    temp_path = "intermediate.mp4"
    num_clips, clip_len, _, height, width = clips.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    # Flatten clips back into individual frames with annotations
    for clip_idx in range(num_clips):
        is_violent = output[clip_idx] >= 0.5  # Threshold to determine "Violence" or "No Violence"
        label = "Violence" if is_violent else "No Violence"
        color = (0, 0, 255) if is_violent else (0, 255, 0)  # Red for violence, green otherwise

        for frame_idx in range(clip_len):
            # Extract and process the frame
            frame = clips[clip_idx, frame_idx].permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
            frame_bgr = cv2.cvtColor((frame).astype(np.uint8), cv2.COLOR_RGB2BGR)

            # Annotate the frame
            cv2.putText(frame_bgr, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame_bgr, (10, 10), (width - 10, height - 10), color, 3)

            # Write the annotated frame to the video
            out.write(frame_bgr)

    out.release()

    subprocess.call(args=f"ffmpeg -y -i {temp_path} -c:v libx264 {output_path}".split(" "))

    if os.path.exists(temp_path):
        os.remove(temp_path)
    print(f"Annotated video saved to {output_path}")


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.conf")
    classifier = ViolenceClassifierInference(config['Inference']['model_path'], config['Inference']['device'])
    datapath = config['Dataset']['dataset_path']
    violence_dir = config['Dataset']['violence_directory']
    nonviolence_dir = config['Dataset']['non_violence_directory']
    video = "NV_112.mp4" #---------change the video u want to see here. If it is of type V_ use violence_dir else use nonviolence_dir in below line
    video_path = os.path.join(os.getcwd(),datapath +os.path.sep + nonviolence_dir + os.path.sep + video)
    clips = get_clips(video_path)
    video_name = os.path.basename(video_path)
    output = classifier.infer(clips)
    print(output)

    # output_folder = "/home/ubuntu/FinalProject/Code/Inference_Output/"
    output_folder = config["Inference"]["output_dir"]
    output_dir = os.path.join(os.getcwd(), output_folder+os.path.sep)
    print(output_dir)
    os.makedirs(output_folder, exist_ok=True)
    stitch_clips_with_annotations(clips, output, output_dir+video_name, fps=30)
