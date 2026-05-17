import torch
import torch.nn as nn
from torchvision import models


class MedicalNet(nn.Module):

    def __init__(self, num_classes=3):
        super(MedicalNet, self).__init__()

        self.model = models.resnet50(pretrained=True)

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)