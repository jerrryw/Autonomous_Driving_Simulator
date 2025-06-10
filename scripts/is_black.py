import cv2
import numpy as np

import cv2
import numpy as np

def classify_traffic_light_color(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for each color
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([35, 255, 255])

    green_lower = np.array([40, 100, 100])
    green_upper = np.array([90, 255, 255])

    # Threshold masks
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Count non-zero pixels in each mask
    red_count = np.count_nonzero(red_mask)
    yellow_count = np.count_nonzero(yellow_mask)
    green_count = np.count_nonzero(green_mask)

    total_pixels = image.shape[0] * image.shape[1]
    max_count = max(red_count, yellow_count, green_count)

    if max_count < 0.02 * total_pixels:
        return "black"  # Mostly unlit
    elif max_count == red_count:
        return "red"
    elif max_count == yellow_count:
        return "yellow"
    elif max_count == green_count:
        return "green"
    else:
        return "unknown"


def is_black_light(image_path, brightness_thresh=40, percent_thresh=0.7):
    """
    Determines if a traffic light image is black (unlit).

    Parameters:
    - image_path: Path to the cropped image.
    - brightness_thresh: Pixel value threshold in V (brightness) channel.
    - percent_thresh: Percentage of dark pixels to consider the light as black.

    Returns:
    - True if black, False otherwise.
    """
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract V (brightness) channel
    v_channel = hsv[:, :, 0]

    # Optional: resize for better visibility
    resized = cv2.resize(v_channel, (v_channel.shape[1]*5, v_channel.shape[0]*5), interpolation=cv2.INTER_NEAREST)

    # Display brightness channel
    cv2.imshow("V Channel (Brightness)", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Create mask of "dark" pixels
    dark_pixels = v_channel < brightness_thresh

    # Calculate ratio of dark pixels
    dark_ratio = np.sum(dark_pixels) / v_channel.size

    # print(f"{image_path}: dark ratio = {dark_ratio:.2f}")

    return dark_ratio > percent_thresh


# Example usage
image_paths = [
    "self_driving/simulator/logs/lights/bot_1749500375_0_451.png",
    "self_driving/simulator/logs/lights/bot_1749500375_0_56a.png",
    "self_driving/simulator/logs/lights/bot_1749500375_0_b3f.png",
]

for path in image_paths:
    result = classify_traffic_light_color(path)
    print(f"{path} is black? {result}")
