import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

# Define a simple CNN
class TrafficDirectionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load the model
model =TrafficDirectionCNN()
model.load_state_dict(torch.load("self_driving/simulator/models/is_traffic_light_model.pth", map_location='cpu'))
model.eval()

# Define transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Define class names (depends on your folder structure)
class_names = ['false', 'true']

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dim: [1, 3, 64, 64]

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

    print(f"Prediction: {class_names[predicted_class]} (Confidence: {confidence:.2f})")

# Example usage:
# python predict.py path/to/image.png
if __name__ == "__main__":
    image_path = "self_driving/simulator/data/light/true/traffic_light_1749585861_0_789.png"
    # image_path = "self_driving/simulator/data/light/false/traffic_light_1749585861_3_38f.png"
    predict(image_path)
