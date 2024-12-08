import torch
from torchvision.transforms.v2 import ToTensor, Compose
from video_resnet import ViolenceClassifier
from torchvision.io import read_video
from torchvision.models.video import R3D_18_Weights

class ViolenceClassifierInference():
    def __init__(self):
        self.classifier = ViolenceClassifier(num_classes=1)
        self.transforms = R3D_18_Weights.DEFAULT.transforms()
        self.classifier.load_state_dict(torch.load("model_resnet.pt"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)
        self.classifier.eval()

    def infer(self, video_frames):
        input = torch.tensor(video_frames)
        input = self.transforms(input)
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

if __name__ == "__main__":
    classifier = ViolenceClassifierInference()
    clips = get_clips("dataset/VID_20241123_155411.mp4")
    output = classifier.infer(clips)
    print(output)
