import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

class TrafficDirectionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 16 * 16, 64)
        self.fc2   = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = TrafficDirectionCNN()
model.load_state_dict(torch.load("self_driving/simulator/models/valid_traffic_light_model.pth", map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def is_valid_traffic_light(image_np: 'np.ndarray') -> bool:
    """Return True if traffic light is facing forward, otherwise False."""
    """
    Given an image path, returns True if the image is classified
    as a valid traffic light (facing front), False otherwise.
    """

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # [1, 3, 64, 64]

    # img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # img_pil = Image.fromarray(img_rgb)
    # x = transform(img_pil).unsqueeze(0)  # shape: [1, 3, 64, 64]

    with torch.no_grad():
        output = model(x)
        prob = torch.sigmoid(output)
        return prob.item() > 0.5

if __name__ == "__main__":
    # image_path = "self_driving/data/light/true/traffic_light_1749585861_0_789.png"
    image_path = "self_driving/data/light/false/traffic_light_1749585861_3_38f.png"
    val = is_valid_traffic_light(image_path)
    print(val)