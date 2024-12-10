import cv2
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn
import os
from PIL import Image

class ViolenceClassifier(nn.Module):
    def __init__(self):
        super(ViolenceClassifier, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.model(x)

model = ViolenceClassifier()
model.load_state_dict(torch.load('./models/static_model.pt', map_location=torch.device('cpu')))
model.eval()

target_layers = [model.model.layer4[-1]]

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

video_path = '../dataset/Violence/V_138.mp4'
output_frames_folder = 'output_frames'
output_gif_path = 'gradcam_output.gif'

os.makedirs(output_frames_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

with GradCAM(model=model, target_layers=target_layers) as cam:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(rgb_img, (224, 224))
        input_tensor = preprocess(img_resized).unsqueeze(0)
        targets = [ClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        visualization = show_cam_on_image(img_resized / 255.0, grayscale_cam, use_rgb=True)
        visualization_resized = cv2.resize(visualization, (frame.shape[1], frame.shape[0]))
        visualization_bgr = cv2.cvtColor(visualization_resized, cv2.COLOR_RGB2BGR)

        frame_path = os.path.join(output_frames_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, visualization_bgr)

        frame_count += 1

cap.release()

frames = [Image.open(os.path.join(output_frames_folder, f)) for f in sorted(os.listdir(output_frames_folder))]
frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
