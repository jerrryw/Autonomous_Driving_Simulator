import os
import cv2
import time
import uuid
import math
import carla
import numpy as np
from ultralytics import YOLO

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
# def classify_traffic_light(image_path):
def classify_traffic_light(img):
    # img = Image.open(image_path).convert("RGB")
    # Step 1: Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 2: Convert NumPy RGB to PIL image
    img_pil = Image.fromarray(img_rgb)
    x = transform(img_pil).unsqueeze(0)  # shape: [1, 3, 64, 64]

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    classes = ['black', 'green', 'red', 'yellow']  # adjust to match your dataset class order
    # return classes[pred], probs.numpy()
    return classes[pred]

# TODO: determine if yolo detected traffic light is facing toward our vehicle
def is_traffic_light(image):
    pass

# global variable
# frame_count = 0
counter = 0
def process_img(image):
    # # Check if timestamps increment by 'sensor_tick' value and number of frames match
    # global frame_count
    # frame_count += 1
    # print(f"Frame {image.frame} at {image.timestamp:.2f} sec")
    # print(f"Captured frame {frame_count} at {image.timestamp:.2f}")

    global counter
    # Convert CARLA image to NumPy
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))

    # drop channel A, RGB
    frame = array[:, :, :3] # [:, :, ::-1]

    # # drop channel A, BGRA -> RGB
    # frame = array[:, :, :3][:, :, ::-1]

    # Draw green trapezoid for route
    # height, width = frame.shape[:2]
    # mid_x         = width // 2
    # bottom_y      = height

    # # Get vehicle yaw (orientation)
    # yaw_deg = vehicle.get_transform().rotation.yaw  # degrees
    # yaw_rad = np.deg2rad(yaw_deg)

    # # Calculate curved offset
    # offset = int(np.sin(yaw_rad) * 100)  # pixel offset left/right

    # # Define trapezoid path points (simulate curve with yaw offset)
    # bottom_width = int(width * 0.5)
    # top_width    = int(width * 0.1)
    # path_length  = int(height * 0.4)
    # top_y        = height - path_length

    # pts = np.array(
    #     [
    #         [mid_x - top_width + offset, top_y],
    #         [mid_x + top_width + offset, top_y],
    #         [mid_x + bottom_width, bottom_y],
    #         [mid_x - bottom_width, bottom_y]
    #     ], dtype=np.int32
    # )

    # Draw curved green path on copied frame (to be YOLO-processed), without permanently affect original frame
    frame_with_path = frame.copy()
    # cv2.fillPoly(frame_with_path, [pts], color=(0, 150, 0))  # green trapezoid

    # Run YOLO on the image with the path
    results   = model(frame_with_path)[0]
    annotated = results.plot()

    def detect_traffic_light_color(crop_bgr):
        # Simple HSV color rules to detect RED / GREEN / YELLOW
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

        # Red (two ranges due to hue wrapping)
        red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 70, 50), (179, 255, 255))
        red  = cv2.bitwise_or(red1, red2)

        # Green
        green = cv2.inRange(hsv, (36, 0, 0), (70, 255, 255))

        # Yellow
        yellow = cv2.inRange(hsv, (15, 0, 0), (36, 255, 255))

        # print("red:", red)

        red_count    = cv2.countNonZero(red)
        green_count  = cv2.countNonZero(green)
        yellow_count = cv2.countNonZero(yellow)

        print("red_count: ", red_count)
        print("green_count: ", green_count)
        print("yellow_count: ", yellow_count)
        print("\n")

        if red_count > green_count and red_count > yellow_count:
            return 'Red'
        elif green_count > red_count and green_count > yellow_count:
            return 'Green'
        elif yellow_count > red_count and yellow_count > green_count:
            return 'Yellow'
        else:
            return 'Unknown'

    def detect_traffic_light_by_circles(crop):
        # height = crop_bgr.shape[0]
        # thirds = height // 3

        # # Split into top, middle, bottom
        # top = crop_bgr[0:thirds, :]
        # middle = crop_bgr[thirds:2*thirds, :]
        # bottom = crop_bgr[2*thirds:, :]

        crop    = frame[y1:y2, x1:x2]
        h, w, _ = crop.shape

        # Divide vertically into three equal parts
        third_h = h // 3

        top_crop    = crop[0:third_h, :]
        middle_crop = crop[third_h:2*third_h, :]
        bottom_crop = crop[2*third_h:h, :]

        # Unique ID per light crop
        uid = f"{int(time.time())}_{idx}_{uuid.uuid4().hex[:3]}"

        # Save all three crop levels
        cv2.imwrite(f"self_driving/simulator/logs/color_lights/top_{uid}.png", top_crop)
        cv2.imwrite(f"self_driving/simulator/logs/color_lights/mid_{uid}.png", middle_crop)
        cv2.imwrite(f"self_driving/simulator/logs/color_lights/bot_{uid}.png", bottom_crop)

        def is_black(img, brightness_thresh=40, percent_thresh=0.7):
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # avg = np.mean(gray)
            # return avg < 60  # Threshold for "black" (tune this if needed)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Extract V (brightness) channel
            v_channel = hsv[:, :, 2]

            # Create mask of "dark" pixels
            dark_pixels = v_channel < brightness_thresh

            # Calculate ratio of dark pixels
            dark_ratio = np.sum(dark_pixels) / v_channel.size

            # print(f"{image_path}: dark ratio = {dark_ratio:.2f}")

            return dark_ratio > percent_thresh

        top_black = is_black(top_crop)
        mid_black = is_black(middle_crop)
        bot_black = is_black(bottom_crop)

        # Apply rules
        if mid_black and bot_black and not top_black:
            return 'Red'
        elif top_black and bot_black and not mid_black:
            return 'Yellow'
        elif top_black and mid_black and not bot_black:
            return 'Green'
        else:
            return 'Unknown'

    # identify all traffic lights in front
    log_lines = []
    for box in results.boxes:
        cls = results.names[int(box.cls)]
        if cls == 'traffic light':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # print("x1:", x1)
            # print("x2:", x2)
            # print("y1:", y1)
            # print("y2:", y2)

            # crop = frame[y1:y2, x1:x2]
            crop = frame[y1:y2, x1:x2]

            filename = os.path.join("self_driving/simulator/logs/traffic_lights", f"traffic_light_{counter}.png")
            cv2.imwrite(filename, crop)
            counter += 1

            # h, w, _ = crop.shape

            # # Divide vertically into three equal parts
            # third_h = h // 3

            # top_crop = crop[0:third_h, :]
            # middle_crop = crop[third_h:2*third_h, :]
            # bottom_crop = crop[2*third_h:h, :]

            # # Optional: visualize or debug
            # cv2.imwrite("self_driving/simulator/logs/color_top_crop.png", top_crop)
            # cv2.imwrite("self_driving/simulator/logs/color_middle_crop.png", middle_crop)
            # cv2.imwrite("self_driving/simulator/logs/color_bottom_crop.png", bottom_crop)


            inferred_state = detect_traffic_light_color(crop)
            # inferred_state = detect_traffic_light_by_circles(crop)

            # Overlay label on annotated frame
            label = f'{inferred_state}'
            # cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
            # cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Validate with CARLA API
            for light in world.get_actors().filter('traffic.traffic_light'):
                if not light.is_alive or not vehicle.is_alive:
                    continue
                try:
                    loc = light.get_transform().location
                    if vehicle.get_location().distance(loc) < 30:
                        true_state = light.state  # carla.TrafficLightState.Red etc.
                        log_line = f'True: {true_state}, Inferred: {inferred_state}\n'
                        log_lines.append(log_line)
                        break
                except RuntimeError:
                    continue

    with open("self_driving/simulator/logs/output.txt", "a") as log_file:
        log_file.writelines(log_lines)


    # # focus on traffic light in front, two at most
    # frame_center_x = frame.shape[1] // 2
    # frame_middle_right_x = frame_center_x + frame.shape[1] // 3  # adjust if needed

    # # Collect traffic lights on front-right side
    # traffic_lights = []

    # for box in results.boxes:
    #     cls = results.names[int(box.cls)]
    #     if cls == 'traffic light':
    #         x1, y1, x2, y2 = map(int, box.xyxy[0])
    #         # print("x1:", x1)
    #         # print("x2:", x2)
    #         # print("y1:", y1)
    #         # print("y2:", y2)

    #         cx = (x1 + x2) // 2
    #         if frame_center_x <= cx <= frame_middle_right_x:
    #             crop = frame[y1:y2, x1:x2]
    #             dist_to_center = abs(cx - frame_center_x)
    #             traffic_lights.append({
    #                 'coords': (x1, y1, x2, y2),
    #                 'crop': crop,
    #                 'cx': cx,
    #                 'distance_to_center': dist_to_center
    #             })

    # # Sort by how close they are to center (more frontal)
    # traffic_lights.sort(key=lambda item: item['distance_to_center'])

    # # Take up to 2 most centered
    # focused_lights = traffic_lights[:2]

    # # for light in focused_lights:
    # for idx, light in enumerate(focused_lights):
    #     x1, y1, x2, y2 = light['coords']
    #     crop = light['crop']

    #     crop = frame[y1:y2, x1:x2]
    #     h, w, _ = crop.shape

    #     # Divide vertically into three equal parts
    #     third_h = h // 3

    #     top_crop    = crop[0:third_h, :]
    #     middle_crop = crop[third_h:2*third_h, :]
    #     bottom_crop = crop[2*third_h:h, :]

    #     # Optional: visualize or debug
    #     # cv2.imwrite("self_driving/simulator/logs/color_lights/top_crop.png", top_crop)
    #     # cv2.imwrite("self_driving/simulator/logs/color_lights/middle_crop.png", middle_crop)
    #     # cv2.imwrite("self_driving/simulator/logs/color_lights/bottom_crop.png", bottom_crop)

    #     # # Unique ID per light crop
    #     # uid = f"{int(time.time())}_{idx}_{uuid.uuid4().hex[:3]}"

    #     # # Save all three crop levels
    #     # cv2.imwrite(f"self_driving/simulator/logs/color_lights/top_{uid}.png", top_crop)
    #     # cv2.imwrite(f"self_driving/simulator/logs/color_lights/mid_{uid}.png", middle_crop)
    #     # cv2.imwrite(f"self_driving/simulator/logs/color_lights/bot_{uid}.png", bottom_crop)

    #     # inferred_state = detect_traffic_light_color(crop)
    #     inferred_state = detect_traffic_light_by_circles(crop)

    #     # inferred_state_top = classify_traffic_light(top_crop)
    #     # inferred_state_mid = classify_traffic_light(middle_crop)
    #     # inferred_state_bot = classify_traffic_light(bottom_crop)

    #     # if (inferred_state_bot and inferred_state_mid) == "black":
    #     #     inferred_state = "Red"
    #     # elif (inferred_state_bot and inferred_state_top) == "black":
    #     #     inferred_state = "Yellow"
    #     # elif (inferred_state_top and inferred_state_mid) == "black":
    #     #     inferred_state = "Green"
    #     # else:
    #     #     inferred_state = "Unknown"

    #     # Draw label
    #     cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
    #     # cv2.putText(annotated, inferred_state, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    #     # Validation (optional)
    #     for carla_light in world.get_actors().filter('traffic.traffic_light'):
    #         if not carla_light.is_alive or not vehicle.is_alive:
    #             continue
    #         try:
    #             # TODO: true_state is incorrect
    #             if vehicle.get_location().distance(carla_light.get_location()) < 60:
    #                 true_state = carla_light.state
    #             # if vehicle.get_location().is_at_traffic_light():
    #             #     true_state = vehicle.get_traffic_light()
    #                 log_line = f"True: {true_state}, Inferred: {inferred_state}\n"
    #                 print(log_line.strip())
    #                 with open("self_driving/simulator/logs/output.txt", "a") as f:
    #                     f.write(log_line)
    #                 break
    #         except RuntimeError:
    #             continue

    # Show window
    # cv2.imshow("results", annotated)

    # Save to video
    video_writer.write(annotated)

def calculate_accuracy(path="self_driving/simulator/logs/output.txt"):
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

    # Optionally, write the result back to the file
    # with open("output.txt", "a") as file:
    #     file.write(f"\n[Summary] Accuracy: {accuracy:.2f}% ({correct}/{total})\n")

if __name__=="__main__":
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)

    # client.load_world('Town01')
    world = client.get_world()

    # Get blueprint library
    blueprint_library = world.get_blueprint_library()

    # Spawn vehicle
    vehicle_bp  = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    # print(spawn_point)

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Autopilot vehicle
    # TODO: replace with your own suggesting route model
    vehicle.set_autopilot(True)

    # Attach RGB camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('fov', '40.0')
    # camera_bp.set_attribute('enable_postprocess_effects', 'True')
    camera_bp.set_attribute('sensor_tick', '0.05') # must match hard-coded fps

    # camera_transform = carla.Transform(carla.Location(x=0.5, z=2.0)) # front of car
    camera_transform = carla.Transform(carla.Location(z=1.5)) # top of car

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Load YOLOv8 model
    # COCO-pretrained
    # TODO: can use yolo11n.pt for more advanced object detection?
    # Currently using yolov8m.pt for better balance of speed and accuracy
    model = YOLO('self_driving/simulator/models/yolo11m.pt')
    # model.predict(verbose=False)

    # Create video output directory and writer
    video_filename = 'self_driving/simulator/logs/yolo_detections.avi'
    fps            = 20
    # frame_size     = (800, 600)  # match image_size_x and y
    frame_size     = (1280, 720)  # match image_size_x and y
    # frame_size     = (1920, 1080)  # match image_size_x and y
    fourcc         = cv2.VideoWriter_fourcc(*'XVID')
    video_writer   = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

    # Start streaming camera
    camera.listen(lambda image: process_img(image))

    # Let simulation run
    time.sleep(10)

    # try:
    #     while True:
    #         time.sleep(10)
    # except KeyboardInterrupt:
    #     print("Stopping Camera...")
    #     camera.stop()

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
