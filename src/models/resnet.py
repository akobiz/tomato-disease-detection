# © 2026 Андрій Кобізь
#  ResNet18 із заміненим останнім шаром (transfer learning
import torch.nn as nn
from torchvision import models


def build_model(num_classes, pretrained=True, freeze_backbone=False):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:

        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)  # новий шар навчається
    return model
