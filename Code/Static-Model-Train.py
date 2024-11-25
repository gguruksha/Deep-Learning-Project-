import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models,transforms
import os
import numpy as np
import configparser


# Define paths - # change the path
ROOT_DIR = os.getcwd()
DATA_DIR = 'dataset'
output_dir = 'models'



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


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Create DataLoader
batch_size = 128
dataset = ViolenceDataset(root_dir=output_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = int(0.2 * len(dataset))
train_dataset,test_dataset = random_split(dataset, [train_size,test_size])

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Load a pre-trained ResNet model
class ViolenceClassifier(nn.Module):
    def __init__(self):
        super(ViolenceClassifier, self).__init__()
        # Load a pre-trained ResNet model
        self.model = models.resnet50(pretrained=True)
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

# Training function
def train_model(model, train_dataloader,test_dataloader ,criterion, optimizer, num_epochs=5):
      # Set the model to training mode

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        met_test = 0
        for images, labels in tqdm(train_dataloader):
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

    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader):
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)

          # Calculate accuracy
          _, predicted = torch.max(outputs.data, 1)
          total_test += labels.size(0)
          correct_test += (predicted == labels).sum().item()


        accuracy_test = 100 * correct_test / total_test
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test - Accuracy: {accuracy_test:.2f}%")

        if accuracy_test > met_test:
          met_test = accuracy_test

          # change the path
          torch.save(model.state_dict(), "static_model.pt")


# Train the model
train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=50)



