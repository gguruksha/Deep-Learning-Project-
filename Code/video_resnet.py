from tqdm import tqdm
from dataloader import get_data_loader
from torchvision.models.video import r3d_18, R3D_18_Weights
from torch import nn
import torch
from torch.nn import BCELoss
from torch import optim
import configparser
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class ViolenceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViolenceClassifier, self).__init__()
        self.num_classes = num_classes

        self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x = self.activation(x)
        return x


def train_model(train_dl, epochs, device):
    model = ViolenceClassifier(num_classes=1)
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.fc.parameters():
        param.requires_grad = True
    model = model.to(device)
    loss_function = BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        pb = tqdm(total=len(train_dl), desc=f"Epoch {epoch+1}")
        total_loss, num_steps = 0, 0
        for img, label in train_dl:
            label = torch.reshape(label, (-1, 1))
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            pb.update(1)
            pb.set_postfix_str(f"Step loss: {loss.item() :.4f}")
            total_loss += loss.item()
            num_steps += 1

        print(f"Epoch {epoch+1} loss: {total_loss / num_steps}")

    torch.save(model.state_dict(), "model_resnet.pt")

def test_model(test_dl, device):
    model = ViolenceClassifier(num_classes=1)
    model.load_state_dict(torch.load("model_resnet.pt"))
    model.to(device)
    model.eval()
    y_true = np.zeros((1,1))
    y_pred = np.zeros((1,1))
    with torch.no_grad():
        pb = tqdm(total=len(test_dl), desc=f"Test")
        for img, label in test_dl:
            label = torch.reshape(label, (-1, 1))
            img, label = img.to(device), label.to(device)
            output = model(img)
            pb.update(1)
            y_true = np.vstack((y_true, label.detach().cpu().numpy()))
            y_pred = np.vstack((y_pred, output.detach().cpu().numpy()))
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    print(f"Accuracy score: {accuracy_score(y_true, y_pred)}")
    print(f"F1 score: {f1_score(y_true, y_pred)}")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.conf")
    epochs = int(config["Training"]["Epochs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, test_dl = get_data_loader(R3D_18_Weights.DEFAULT.transforms(), clip_len=16)
    # train_model(train_dl, epochs, device)
    test_model(test_dl, device)