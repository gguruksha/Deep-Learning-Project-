import configparser
import os
import pandas as pd
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import ToTensor
from torchvision.io import read_video

def prepare_file_paths(data_dir, violence_dir, nonviolence_dir):
    nonviolence_path = data_dir + os.path.sep + nonviolence_dir
    violence_path = data_dir + os.path.sep + violence_dir
    paths = []
    labels = []
    for f in os.listdir(nonviolence_path):
        paths.append(nonviolence_path + os.path.sep + f)
        labels.append(0)

    for f in os.listdir(violence_path):
        paths.append(violence_path + os.path.sep + f)
        labels.append(1)

    files_df = pd.DataFrame({'path': paths, 'label': labels})
    return files_df

def train_test_paths(all_paths, test_split, random_seed):
    train_paths = all_paths.sample(frac=(1 - float(test_split)), random_state=random_seed)
    test_paths = all_paths.drop(train_paths.index)
    return train_paths, test_paths

class ViolentVideos(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.files_labels = pd.read_csv(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.files_labels)

    def __getitem__(self, idx):
        video = read_video(self.files_labels['path'][idx])
        label = self.files_labels['label'][idx]
        if self.transform:
            video = self.transform(video)
        return video, label


def get_data_loader():
    config = configparser.ConfigParser()
    config.read('config.conf')
    test_file_path = config.get('Dataset', 'dataset_path') + os.path.sep + 'test.csv'
    train_file_path = config.get('Dataset', 'dataset_path') + os.path.sep + 'train.csv'

    if (not os.path.exists(test_file_path)) or (not os.path.exists(train_file_path)):
        paths = prepare_file_paths(
            config.get('Dataset', 'dataset_path'),
            config.get('Dataset', 'violence_directory'),
            config.get('Dataset', 'non_violence_directory')
        )
        train, test = train_test_paths(
            paths,
            config.get('Dataset', 'test_split'),
            int(config.get('Dataset', 'random_seed'))
        )
        train.to_csv(train_file_path, index=False)
        test.to_csv(test_file_path, index=False)

    train_dataset = ViolentVideos(train_file_path)
    test_dataset = ViolentVideos(test_file_path)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader
