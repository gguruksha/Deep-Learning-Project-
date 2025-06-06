import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models,transforms
import os

# Define paths
ROOT_DIR = os.getcwd()
os.chdir("..")
DATA_DIR = os.getcwd() + '/Data'

output_dir = DATA_DIR + '/Preprocessed_Frames'
# os.makedirs(output_dir, exist_ok=True)

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
    transforms.ToTensor(),               # Convert image to Tensor
    transforms.Normalize([0.5], [0.5])   # Normalize to range [-1, 1]
])

# Create DataLoader
dataset = ViolenceDataset(root_dir=output_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print("DataLoad Complete")



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
def train_model(model, data_loader, criterion, optimizer, num_epochs=5):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(data_loader):
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

        epoch_loss = running_loss / len(data_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")


# Train the model
train_model(model, data_loader, criterion, optimizer, num_epochs=10)



