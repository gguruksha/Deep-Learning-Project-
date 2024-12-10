import os
import torch
from run_inference import ViolenceClassifierInference, get_clips, stitch_clips_with_annotations
from video_resnet import get_data_loader
from torchvision.models.video import R3D_18_Weights
import configparser


def process_test_dataloader(test_dataloader, output_folder, model):
    """
    Process the test dataloader and annotate each video.

    Args:
        test_dataloader: DataLoader object for the test dataset.
        output_folder: Directory where annotated videos will be saved.
        model: The inference model.
    """

    for batch_idx, (videos, labels) in enumerate(test_dataloader):
        print(batch_idx)
        for idx in range(videos.shape[0]):
            video_path = test_dataloader.dataset.files_labels['path'][batch_idx * test_dataloader.batch_size + idx]
            print(video_path)

            # Get clips from video
            clips = get_clips(video_path)

            # Run inference
            output = model.infer(clips)

            # Generate annotated video
            video_name = os.path.basename(video_path)
            output_path = os.path.join(output_folder, f"annotated_{video_name}")
            stitch_clips_with_annotations(clips, output, output_path)



if __name__ == '__main__':
    # Initialize model and dataloader
    config = configparser.ConfigParser()
    config.read("config.conf")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # classifier = ViolenceClassifierInference("model_resnet.pt", device)
    classifier = ViolenceClassifierInference(config['Inference']['model_path'], config['Inference']['device'])

    # Load dataloaders
    train_dl, test_dl = get_data_loader(R3D_18_Weights.DEFAULT.transforms(), clip_len=16)
    # print(test_dl)

    # Output folder for annotated videos
    # output_folder = "./annotated_videos"
    output_folder = config["Inference"]["output_dir"]
    output_dir = os.path.join(os.getcwd(), output_folder + os.path.sep)
    # print(output_dir)
    os.makedirs(output_folder, exist_ok=True)

    # Process the test dataloader
    process_test_dataloader(test_dl, output_dir, classifier)
