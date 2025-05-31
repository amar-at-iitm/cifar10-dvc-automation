import logging
import os
import random
import shutil
import yaml
from logger import setup_logger

# Setup logging
setup_logger('logs/data_preparation.log')
logger = logging.getLogger('data_preparation')

# Load configuration
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# Prepare data splits
def create_split(data, split_ratio):
    split_size = int(len(data) * split_ratio)
    return data[:split_size], data[split_size:]

def prepare_data():
    random.seed(config['seed'])
    version = config['pull_data']['version']
    partition_dir = f'data/partitions/{version}'
    output_dir = 'data/splits'

    # Clean up previous splits
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_images = []
    for class_name in os.listdir(partition_dir):
        class_dir = os.path.join(partition_dir, class_name)
        if os.path.isdir(class_dir):
            images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
            all_images.extend(images)

    random.shuffle(all_images)

    train, val = create_split(all_images, config['split']['train_ratio'])
    val, test = create_split(val, config['split']['val_ratio'] / (config['split']['val_ratio'] + config['split']['test_ratio']))

    for split_name, split_data in zip(['train', 'val', 'test'], [train, val, test]):
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for img in split_data:
            class_name = os.path.basename(os.path.dirname(img))
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy(img, class_dir)
        logger.info(f'Created {split_name} set with {len(split_data)} images')

prepare_data()
