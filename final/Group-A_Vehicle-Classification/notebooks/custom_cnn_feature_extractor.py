import torch
import torch.nn as nn
from torchvision import models

class CustomCNNFeatureExtractor(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNNFeatureExtractor, self).__init__()

        # Define the convolutional layers with increased complexity
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),  # Increased number of neurons
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Additional dropout layer
            nn.Linear(512, num_classes)
        )

        # Option to extract features before the final classification layer
        self.feature_extract = False

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        if self.feature_extract:
            return x
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Example usage for feature extraction
    model = CustomCNNFeatureExtractor(num_classes=10)
    x = torch.randn(64, 3, 224, 224)
    features = model(x)
    print(features.size())  # Size of extracted feature vector
