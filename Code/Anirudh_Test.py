from torch.utils.data import Dataset, DataLoader
import glob


class ViolenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Get list of all image paths
        self.violence_images = glob.glob(os.path.join(root_dir, 'Violence', '*.jpg'))
        self.nonviolence_images = glob.glob(os.path.join(root_dir, 'NonViolence', '*.jpg'))
        # Combine lists with labels
        self.image_paths = self.violence_images + self.nonviolence_images
        self.labels = [1] * len(self.violence_images) + [0] * len(self.nonviolence_images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


# Create DataLoader
dataset = ViolenceDataset(root_dir=output_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Load a pre-trained ResNet model
class ViolenceClassifier(nn.Module):
    def __init__(self):
        super(ViolenceClassifier, self).__init__()
        # Load a pre-trained ResNet model
        self.model = models.resnet18(pretrained=True)
        # Modify the final layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)  # 2 classes: violence and non-violence

    def forward(self, x):
        return self.model(x)

# Initialize model, loss, and optimizer
model = ViolenceClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from tqdm import tqdm  # for progress bar
import torch.nn.functional as F

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


