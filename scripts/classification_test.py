import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Define the CNN architecture (same as training) ---
class TrafficLightCNN(nn.Module):
    def __init__(self):
        super(TrafficLightCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 4)  # 4 classes: red, yellow, green, black

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # Output: (B, 16, 32, 32)
        x = self.pool(F.relu(self.conv2(x)))   # Output: (B, 32, 16, 16)
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- 2. Load model and weights ---
model = TrafficLightCNN()
model.load_state_dict(torch.load("self_driving/simulator/models/traffic_light_color_model.pth", map_location=torch.device('cpu')))
model.eval()

# --- 3. Define preprocessing (same as training) ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# --- 4. Predict on new image ---
def classify_traffic_light(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # shape: [1, 3, 64, 64]

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    classes = ['black', 'green', 'red', 'yellow']  # adjust to match your dataset class order
    return classes[pred], probs.numpy()

# --- 5. Example usage ---
label1, confidence1 = classify_traffic_light("self_driving/data/color/black/bot_1749500375_0_8eb.png")
label2, confidence2 = classify_traffic_light("self_driving/data/color/green/bot_1749430935_0_002.png")
label3, confidence3 = classify_traffic_light("self_driving/data/color/red/top_1749431102_0_6b7.png")
label4, confidence4 = classify_traffic_light("self_driving/data/color/yellow/mid_1749431426_0_0eb.png")
print(f"Predicted: {label1}, Confidence: {confidence1}")
print(f"Predicted: {label2}, Confidence: {confidence2}")
print(f"Predicted: {label3}, Confidence: {confidence3}")
print(f"Predicted: {label4}, Confidence: {confidence4}")
