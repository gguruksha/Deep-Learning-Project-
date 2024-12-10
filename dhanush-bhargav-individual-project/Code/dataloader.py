import configparser
import os
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import ToTensor, Compose
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
    def __init__(self, annotations_file, transform=None, clip_len=32):
        self.files_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.clip_len = clip_len

    def __len__(self):
        return len(self.files_labels)

    def __getitem__(self, idx):
        video, audio, metadata = read_video(self.files_labels['path'][idx], output_format="TCHW", pts_unit='sec')
        label = self.files_labels['label'][idx]
        max_seek = video.shape[0] - self.clip_len
        start = random.randint(0, max_seek)
        vid = video[start:start + self.clip_len]

        if self.transform:
            vid = self.transform(vid)
        return vid, torch.tensor(label, dtype=torch.float32)


def get_data_loader(transforms, clip_len=32):
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

    train_dataset = ViolentVideos(train_file_path, transform=transforms, clip_len=clip_len)
    test_dataset = ViolentVideos(test_file_path, transform=transforms, clip_len=clip_len)

    video, label = train_dataset.__getitem__(1181)

    train_dataloader = DataLoader(train_dataset, batch_size=int(config.get('Dataset', 'batch_size')), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=int(config.get('Dataset', 'batch_size')), shuffle=True)

    return train_dataloader, test_dataloader

if __name__ == '__main__':
    train_loader, test_loader = get_data_loader(Compose([ToTensor()]))
