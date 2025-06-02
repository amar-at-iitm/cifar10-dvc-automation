import itertools
import logging
import yaml
import torch
import os
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from train_model import CNN
from logger import setup_logger

# Setup logging
setup_logger('logs/task5.log')
logger = logging.getLogger('task5')

# Loading configuration
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# Checking device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Dataset combinations
datasets_list = ["v1", "v2", "v3"]
dataset_combinations = []
for r in range(1, len(datasets_list) + 1):
    combinations = ["+".join(comb) for comb in itertools.combinations(datasets_list, r)]
    dataset_combinations.extend(combinations)

# Identify hard-to-learn images
def identify_hard_to_learn():
    hard_to_learn_images = []
    for combination in dataset_combinations:
        logger.info(f'Processing combination: {combination}')

        # Loading model
        model = CNN().to(device)
        model_path = f'model/model_{combination}.pth'
        if not os.path.exists(model_path):
            logger.warning(f'Model not found: {model_path}')
            continue
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Loading test data
        transform = transforms.Compose([transforms.ToTensor()])
        test_data = datasets.ImageFolder(f'data/splits/test_{combination}', transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['training']['batch_size'], shuffle=False)

        misclassified = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                for img, pred, label in zip(images, preds, labels):
                    if pred != label:
                        misclassified.append((img.cpu(), label.cpu().item(), pred.cpu().item()))
        hard_to_learn_images.extend(misclassified)

    # Saves hard-to-learn images and their misclassification details
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(hard_to_learn_images, columns=['Image', 'True Label', 'Predicted Label'])
    df.to_csv('results/hard_to_learn.csv', index=False)
    logger.info('Hard-to-learn images and misclassification details saved to results/hard_to_learn.csv')

identify_hard_to_learn()
