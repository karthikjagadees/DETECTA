# models/classifier.py
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=69, pretrained=True):
        super(ResNet50Classifier, self).__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        in_features = self.backbone.fc.in_features
        
        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, x):
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)