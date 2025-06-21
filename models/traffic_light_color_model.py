import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

class TrafficLightCNN(nn.Module):
    def __init__(self):
        super(TrafficLightCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 16 * 16, 64)
        self.fc2   = nn.Linear(64, 4)  # 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, 32, 32)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, 16, 16)
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Transforms
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

# Dataset
dataset    = datasets.ImageFolder('self_driving/data/color', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, loss, optimizer
model     = TrafficLightCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):
    total_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{10}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'self_driving/simulator/models/traffic_light_color_model.pth')
