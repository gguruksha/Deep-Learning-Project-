
import os
import cv2
import torch
from torchvision import transforms
from torch.nn.functional import softmax

def load_model(model_path):
    """
    Loads the pre-trained model from the given path.
    :param model_path: Path to the .pt model file.
    :return: Loaded PyTorch model.
    """
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()  # Set model to evaluation mode
    return model

def extract_frames_from_video(video_path, frame_size=(224, 224), batch_size=16):
    """
    Extract frames from a video and divides them into batches.
    :param video_path: Path to the video file.
    :param frame_size: Resize dimensions for the frames.
    :param batch_size: Number of frames per batch.
    :return: List of frame batches (each batch is a tensor of shape [batch_size, 3, H, W]).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(frame_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ensure the frame has 3 channels (convert grayscale to RGB if necessary)
        if len(frame.shape) == 2 or frame.shape[2] == 1:  # Grayscale case
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3:  # Already RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(transform(frame))

    cap.release()

    # Ensure frames are divisible by batch_size and discard extra frames
    num_batches = len(frames) // batch_size
    frames = frames[:num_batches * batch_size]

    # Divide frames into batches
    frame_batches = [
        torch.stack(frames[i:i + batch_size]) for i in range(0, len(frames), batch_size)
    ]
    return frame_batches


def classify_video(video_path, model_path, batch_size=16):
    """
    Classify a video as "Violent" or "Non-Violent" based on frame batches.
    :param video_path: Path to the video file.
    :param model_path: Path to the trained model.
    :param batch_size: Number of frames per batch.
    :return: "Violent" or "Non-Violent" classification.
    """
    # Load model
    model = load_model(model_path)

    # Extract frame batches
    frame_batches = extract_frames_from_video(video_path, batch_size=batch_size)

    # Perform inference on each batch
    for batch in frame_batches:
        batch = batch.to(torch.device("cpu"))  # Move batch to CPU (or GPU if available)
        with torch.no_grad():
            outputs = model(batch)
            probabilities = softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        # If any batch is classified as "Violent" (label 1), classify the video as "Violent"
        if 1 in predictions:
            return "Violent"

    # If no batch is classified as "Violent", classify the video as "Non-Violent"
    return "Non-Violent"
