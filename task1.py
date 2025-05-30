import os
import numpy as np
import torchvision.datasets as datasets
from tqdm import tqdm
from PIL import Image

# Defining folders
dataset_folder = "raw_data"
output_folder = "cifar10_images"
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Creating subfolders for each class
os.makedirs(output_folder, exist_ok=True)
for class_name in classes:
    os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

# Downloading Train and Test sets separately
trainset = datasets.CIFAR10(root=dataset_folder, train=True, download=True)
testset = datasets.CIFAR10(root=dataset_folder, train=False, download=True)

# Combining train and test datasets
full_dataset = np.concatenate((trainset.data, testset.data), axis=0)  # Merges image arrays
full_labels = trainset.targets + testset.targets  # Merges labels

# Function to save images
def save_images(images, labels):
    for idx in tqdm(range(len(images)), desc="Saving Images"):
        image = images[idx]
        label = labels[idx]
        class_name = classes[label]

        # Defines image path (numbering all images sequentially)
        image_path = os.path.join(output_folder, class_name, f"img_{idx}.png")

        # Converts and saves image
        image = Image.fromarray(np.array(image))
        image.save(image_path)

# Save all 60,000 images
save_images(full_dataset, full_labels)

print("All 60,000 images extracted and saved successfully!")
