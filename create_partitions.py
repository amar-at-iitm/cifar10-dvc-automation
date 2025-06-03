import os
import shutil
import random

# Path to the processed images (from Task 1)
processed_data_path = "cifar10_images"  # Updated from raw_data
partition_path = "data/partitions"

# Create partition directory if not exists
os.makedirs(partition_path, exist_ok=True)

# Collect all images with their class names
all_images = []
for class_name in os.listdir(processed_data_path):
    class_dir = os.path.join(processed_data_path, class_name)
    if os.path.isdir(class_dir):  # Ensure it's a directory
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
        all_images.extend(images)

# Shuffle the images randomly
random.shuffle(all_images)

# Split into three partitions (20,000 images each)
v1, v2, v3 = all_images[:20000], all_images[20000:40000], all_images[40000:]

# Function to copy images into partition folders
def create_partition(version, images):
    partition_dir = os.path.join(partition_path, version)
    os.makedirs(partition_dir, exist_ok=True)
    
    for img_path in images:
        class_name = os.path.basename(os.path.dirname(img_path))  # Get class name
        class_dir = os.path.join(partition_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        shutil.copy(img_path, class_dir)  # Copy image to partition

# Create the partitions
create_partition("v1", v1)
create_partition("v2", v2)
create_partition("v3", v3)

print("Partitions created successfully in 'data/partitions/' .")
