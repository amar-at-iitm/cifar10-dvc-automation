import logging
import yaml
import torch
import os
import json
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from logger import setup_logger
from train_model import CNN  # Importing the CNN model

# Setup logging
setup_logger('logs/evaluate.log')
logger = logging.getLogger('evaluate')

# Load configuration
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load('model/model.pth'))
model.eval()

# Evaluation function
def evaluate():
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.ImageFolder('data/splits/test', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['training']['batch_size'], shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f'Overall Accuracy: {accuracy * 100:.2f}%')

    # Extracts values from config
    seed = config['seed']
    version = config['pull_data']['version'].replace('+', '_')

    # Uses formatted filenames from config.yaml
    report_filename = config['evaluation']['save_report'].format(version=version, seed=seed)
    conf_matrix_filename = config['evaluation']['confusion_matrix'].format(version=version, seed=seed)

    # Ensures directories exist
    os.makedirs(os.path.dirname(report_filename), exist_ok=True)
    os.makedirs(os.path.dirname(conf_matrix_filename), exist_ok=True)

    
    # Generates a classification report with class names and output as a dictionary
    report = classification_report(all_labels, all_preds, target_names=test_data.classes, output_dict=True)
    logger.info(f'Classification Report:\n{json.dumps(report, indent=4)}')

    # Saves the evaluation report as a JSON file
    evaluation_report = {
        "accuracy": accuracy,
        "classification_report": report
    }
    # Saves evaluation report   
    with open(report_filename, 'w') as json_file:
        json.dump(evaluation_report, json_file)
    logger.info(f'Evaluation report saved at {report_filename}')


    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_data.classes, yticklabels=test_data.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Saves confusion matrix
    plt.savefig(conf_matrix_filename)
    logger.info(f"Confusion matrix saved at {conf_matrix_filename}")

evaluate()
