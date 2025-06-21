import os
import cv2
import time
import uuid
import math
import carla
import inspect
import numpy as np
from PIL import Image
from ultralytics import YOLO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class TrafficLightColorModel(nn.Module):
    def __init__(self):
        super(TrafficLightColorModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 16 * 16, 64)
        self.fc2   = nn.Linear(64, 4)  # 4 classes: red, yellow, green, black

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output: (B, 16, 32, 32)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (B, 32, 16, 16)
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

color_model = TrafficLightColorModel()
color_model.load_state_dict(torch.load("self_driving/simulator/models/traffic_light_color_model.pth", map_location=torch.device('cpu')))
color_model.eval()

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

def classify_traffic_light(image):
    # Assume top_crop is a NumPy array from OpenCV (BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # Convert to PyTorch tensor with batch dimension
    x = transform(image_pil).unsqueeze(0) # shape: [1, 3, 64, 64]

    with torch.no_grad():
        outputs = color_model(x)

        if isinstance(outputs, (list, tuple)): outputs = outputs[0]  # Unpack the tensor if wrapped in a list/tuple

        probs = torch.softmax(outputs, dim=1)
        pred  = torch.argmax(probs, dim=1).item()

    classes = ['black', 'green', 'red', 'yellow']  # adjust to match your dataset class order
    return classes[pred], probs.numpy()

# -----------------------------------------------------------------------------------------------------------------------
class ValidTrafficLightModel(nn.Module):
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

# Load the model
valid_light_model = ValidTrafficLightModel()
valid_light_model.load_state_dict(torch.load("self_driving/simulator/models/valid_traffic_light_model.pth", map_location='cpu'))
valid_light_model.eval()

# Define transforms (must match training)
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

# Determine if yolo detected traffic light is facing toward our vehicle
def is_valid_traffic_light(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    x         = transform(image_pil).unsqueeze(0)  # shape: [1, 3, 64, 64]

    with torch.no_grad():
        output = valid_light_model(x)
        prob   = torch.sigmoid(output)
        return prob.item() > 0.5

# -----------------------------------------------------------------------------------------------------------------------
# global variable
# frame_count = 0 # for sensor tick
counter     = 0 # for traffic light image capture
def process_image(image):
    # Check if timestamps increment by 'sensor_tick' value and number of frames match
    # global frame_count
    # frame_count += 1
    # print(f"Frame {image.frame} at {image.timestamp:.2f} sec")
    # print(f"Captured frame {frame_count} at {image.timestamp:.2f}")

    # Convert CARLA image to NumPy
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))

    # drop channel A, RGB
    frame = array[:, :, :3] # [:, :, ::-1]

    # # drop channel A, BGRA -> RGB
    # frame = array[:, :, :3][:, :, ::-1]

    # TODO: Draw green trapezoid for route suggestion
    # Draw curved green path on copied frame (to be YOLO-processed), without permanently affect original frame

    frame_with_path = frame.copy()

    # Run YOLO on the image with the path
    results   = yolo_model(frame_with_path)[0]
    annotated = results.plot()

    # identify all traffic lights in front
    log_lines = []
    for idx, box in enumerate(results.boxes):
        cls = results.names[int(box.cls)]
        if cls == 'traffic light':
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = frame[y1:y2, x1:x2]

            # print("idx =", idx)
            # print("-----------------is_valid_traffic_light(crop)-----------------", is_valid_traffic_light(crop))

            # if (is_valid_traffic_light(crop)):
            #     # Take pictures of traffic lights
            #     global counter
            #     uid = f"{int(time.time())}_{counter}_{uuid.uuid4().hex[:3]}"
            #     filename = os.path.join("self_driving/simulator/logs/traffic_lights", f"traffic_light_{uid}.png")
            #     cv2.imwrite(filename, crop)
            #     counter += 1

            if (is_valid_traffic_light(crop)):

                height, width, _ = crop.shape

                # Divide vertically into three equal parts
                div_parts = height // 3

                top_crop = crop[0:div_parts          , :]
                mid_crop = crop[div_parts:2*div_parts, :]
                bot_crop = crop[2*div_parts:height   , :]

                # # Unique ID per light crop
                # uid = f"{int(time.time())}_{idx}_{uuid.uuid4().hex[:3]}"

                # # Save all three crop levels
                # cv2.imwrite(f"self_driving/simulator/logs/color_lights/top_{uid}.png", top_crop)
                # cv2.imwrite(f"self_driving/simulator/logs/color_lights/mid_{uid}.png", mid_crop)
                # cv2.imwrite(f"self_driving/simulator/logs/color_lights/bot_{uid}.png", bot_crop)

                inferred_top = classify_traffic_light(top_crop)
                inferred_mid = classify_traffic_light(mid_crop)
                inferred_bot = classify_traffic_light(bot_crop)

                if (inferred_bot and inferred_mid) == "black":
                    inferred_state = "red"
                elif (inferred_bot and inferred_top) == "black":
                    inferred_state = "yellow"
                elif (inferred_top and inferred_mid) == "black":
                    inferred_state = "green"
                else:
                    inferred_state = "unknown"

                # Overlay label on annotated frame
                label = f'{inferred_state}'
                # cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
                # cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Validate with CARLA API
                # for light in world.get_actors().filter('traffic.traffic_light'):
                #     if not light.is_alive or not vehicle.is_alive:
                #         continue
                #     try:
                #         loc = light.get_transform().location
                #         if vehicle.get_location().distance(loc) < 30:
                #             true_state = light.state  # carla.TrafficLightState.Red etc.
                #             log_line = f'True: {true_state}, Inferred: {inferred_state}\n'
                #             log_lines.append(log_line)
                #             break
                #     except RuntimeError:
                #         continue

                print("vehicle.is_at_traffic_light() = ", vehicle.is_at_traffic_light())

                # TODO: true_state is incorrect
                if vehicle.is_at_traffic_light():
                    traffic_light = vehicle.get_traffic_light()
                    print("Reached Line", inspect.currentframe().f_lineno)

                    if traffic_light and traffic_light.is_alive:
                        true_state = traffic_light.state
                        log_line = f"True: {true_state}, Inferred: {inferred_state}\n"
                        log_lines.append(log_line)

    print("Reached Line", inspect.currentframe().f_lineno)
    # exit(1)

    with open("self_driving/simulator/logs/output.txt", "a") as log_file:
        log_file.writelines(log_lines)

    # Show window
    # cv2.imshow("results", annotated)

    # Save to video
    video_writer.write(annotated)

def calculate_accuracy(path):
    total   = 0
    correct = 0

    with open(path, 'r') as file:
        for line in file:
            total += 1
            true_val     = line.split("True:")[1].split(",")[0].strip().upper()
            inferred_val = line.split("Inferred:")[1].strip().upper()
            if true_val == inferred_val:
                correct += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Total Samples      : {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy           : {accuracy:.2f}%")

if __name__=="__main__":
    # Connect to CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)

    # print(client.get_available_maps())

    # world = client.load_world('Town10HD_Opt')
    # world = client.reload_world()
    world = client.get_world()

    # Get blueprint library
    blueprint_library = world.get_blueprint_library()

    # Spawn vehicle
    vehicle_bp  = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = world.get_map().get_spawn_points()[1]

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Autopilot vehicle
    # TODO: replace with your own suggesting route model
    vehicle.set_autopilot(False)

    # Attach RGB camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('fov', '40.0')
    camera_bp.set_attribute('sensor_tick', '0.05') # must match hard-coded fps

    # camera_transform = carla.Transform(carla.Location(x=0.5, z=2.0)) # front of car
    camera_transform = carla.Transform(carla.Location(z=1.5))        # top of car

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Load YOLO model
    # COCO-pretrained
    # default used yolov8n.pt
    yolo_model = YOLO('self_driving/simulator/models/yolo11m.pt')

    # Create video output directory and writer
    video_filename = 'self_driving/simulator/logs/yolo_detections.avi'
    fps            = 20
    frame_size     = (1280, 720)  # match image_size_x and y, other options include (800, 600), (1920, 1080)
    fourcc         = cv2.VideoWriter_fourcc(*'XVID')
    video_writer   = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

    # Start streaming camera
    camera.listen(lambda image: process_image(image))

    # Let simulation run
    time.sleep(10)

    camera.stop()
    time.sleep(0.5)
    print("\nCleaning up...")
    vehicle.destroy()
    print("    - All Vehicles Destroyed")
    camera.destroy()
    print("    - Cameras Destroyed")
    video_writer.release()
    print("    - Video Output to yolo_detections.avi")
    cv2.destroyAllWindows()
    print("    - Closing all cv2 windows")
    print("Finished Cleaning.")

    print("\nCalculated Results")
    calculate_accuracy("self_driving/simulator/logs/output.txt")
