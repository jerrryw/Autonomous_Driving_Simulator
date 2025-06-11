import os
import shutil
import random
from pathlib import Path

# Paths
input_dir = Path("self_driving/simulator/data/light")        # original dataset with true/false folders
output_dir = Path("self_driving/simulator/data/light_split") # new directory to hold train/val split
train_ratio = 0.8                  # 80% for training

# Classes (true/false)
classes = ["true", "false"]

# Create train/val folders
for split in ["train", "valid"]:
    for cls in classes:
        (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

# Split each class
for cls in classes:
    images = list((input_dir / cls).glob("*.*"))  # *.png or *.jpg
    random.shuffle(images)

    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Copy files
    for img in train_images:
        shutil.copy(img, output_dir / "train" / cls / img.name)

    for img in val_images:
        shutil.copy(img, output_dir / "valid" / cls / img.name)

print("Dataset split complete.")
