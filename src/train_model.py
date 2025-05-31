import logging
import yaml
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from logger import setup_logger

# Setup logging
setup_logger('logs/train_model.log')
logger = logging.getLogger('train_model')

# Load configuration
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# Extract training parameters
batch_size = config['training']['batch_size']
epochs = config['training']['epochs']
learning_rate = config['training']['learning_rate']
conv_layers = config['training']['conv_layers']
conv_filters = config['training']['conv_filters']
kernel_sizes = config['training']['kernel_sizes']
dropout = config['training']['dropout']

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        conv_layer_list = []
        in_channels = 3

        for i in range(conv_layers):  # Use the conv_layers from the config
            out_channels = conv_filters[i]
            kernel_size = kernel_sizes[i]
            conv_layer_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
            conv_layer_list.append(nn.ReLU())
            conv_layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layer_list)
        self.fc1 = nn.Linear(conv_filters[-1] * 4 * 4, config['training']['fc_units'])
        self.fc2 = nn.Linear(config['training']['fc_units'], 10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training function
def train():
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.ImageFolder('data/splits/train', transform=transform)
    val_data = datasets.ImageFolder('data/splits/val', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Ensure the model directory exists
    os.makedirs('model', exist_ok=True)
    
    torch.save(model.state_dict(), 'model/model.pth')
    logger.info('Model training completed.')

train()
