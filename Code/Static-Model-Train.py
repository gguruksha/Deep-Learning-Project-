import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import os
import numpy as np
import configparser
from tqdm import tqdm  # for progress bar
import torch.nn.functional as F

# Define paths - change the path accordingly
ROOT_DIR = os.getcwd()
config = configparser.ConfigParser()
config.read('config.conf')

DATA_DIR = os.path.join(config.get('Frames', 'frames_dataset_path'), '')
violence_path = os.path.join(DATA_DIR, config.get('Frames', 'violence_frames_dir'))
non_violence_path = os.path.join(DATA_DIR, config.get('Frames', 'non_violence_frames_dir'))
model_op = config.get('StaticModel', 'models_path')
os.makedirs(model_op, exist_ok=True)

class ViolenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get list of all image paths
        self.violence_images = glob.glob(os.path.join(root_dir, config.get('Frames', 'violence_frames_dir'), '*.jpg'))
        self.nonviolence_images = glob.glob(os.path.join(root_dir, config.get('Frames', 'non_violence_frames_dir'), '*.jpg'))

        # Combine lists with labels
        self.image_paths = self.violence_images + self.nonviolence_images
        self.labels = [1] * len(self.violence_images) + [0] * len(self.nonviolence_images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Create DataLoader
batch_size = 128
dataset = ViolenceDataset(root_dir=DATA_DIR, transform=transform)
train_size = int(0.7 * len(dataset))
test_size = int(0.2 * len(dataset))
val_size = len(dataset) - (train_size + test_size)

assert train_size + test_size + val_size == len(dataset)

train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load a pre-trained ResNet model
class ViolenceClassifier(nn.Module):
    def __init__(self):
        super(ViolenceClassifier, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.model(x)

# Initialize model, loss, and optimizer
model = ViolenceClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training function
def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=5):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} - Train'):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Evaluate on test set
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_accuracy = 100 * correct_test / total_test
        print(f"Test Accuracy: {test_accuracy:.2f}%")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), os.path.join(model_op, 'static_model.pt'))

        print(f"Best Test Accuracy So Far: {best_accuracy:.2f}%")

def test_model(model_path, test_dataloader):
    model = ViolenceClassifier()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    accuracy_test = 100 * correct_test / total_test
    print(f"Final Test Accuracy: {accuracy_test:.2f}%")

# Train the model
train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=3)

# Test the model
model_path = os.path.join(model_op, 'static_model.pt')
test_model(model_path, val_dataloader)
