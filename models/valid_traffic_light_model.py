import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Define a simple CNN
class TrafficDirectionCNN(nn.Module):
    def __init__(self):
        super(TrafficDirectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 16 * 16, 64)
        self.fc2   = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 16, 16]
        x = x.view(-1, 32 * 16 * 16)          # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                       # Final output logit (no sigmoid)
        return x

# Define a simple CNN
# class TrafficDirectionCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential\
#         (
#             nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(32 * 16 * 16, 64), nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x):
#         return self.model(x)

# Define data transforms
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

# Load data
train_data   = datasets.ImageFolder("self_driving/data/light_split/train", transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

valid_data   = datasets.ImageFolder("self_driving/data/light_split/valid", transform=transform)
valid_loader = DataLoader(valid_data, batch_size=32)

model     = TrafficDirectionCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model     = model.to(device)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        labels = labels.float().unsqueeze(1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Train Loss: {total_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            labels = labels.float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds.float() == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    print(f"  → Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "self_driving/simulator/models/valid_traffic_light_model.pth")
