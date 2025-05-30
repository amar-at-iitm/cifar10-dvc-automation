import logging
import yaml
import os
import shutil
from logger import setup_logger

# Setup logging
setup_logger('logs/pull_data.log')
logger = logging.getLogger('pull_data')

# Load configuration
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# Get dataset version(s)
version = config['pull_data']['version']
version_list = version.split('+')  # Split in case of multiple versions

# Ensure partitions directory exists
partition_dir = "data/partitions"
os.makedirs(partition_dir, exist_ok=True)

# Create a combined folder if multiple versions are specified
combined_version_dir = os.path.join(partition_dir, version)
if len(version_list) > 1:
    os.makedirs(combined_version_dir, exist_ok=True)

try:
    for v in version_list:
        logger.info(f'Pulling data version: {v}')
        os.system(f'dvc pull data/partitions/{v}')

        # If multiple versions, merge them into one directory
        if len(version_list) > 1:
            src_dir = os.path.join(partition_dir, v)
            for item in os.listdir(src_dir):
                src_path = os.path.join(src_dir, item)
                dest_path = os.path.join(combined_version_dir, item)
                
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)  # Merge class folders
                else:
                    shutil.copy2(src_path, dest_path)  # Copy files

    logger.info(f'Successfully pulled {version}')
except Exception as e:
    logger.error(f'Failed to pull data: {e}')
