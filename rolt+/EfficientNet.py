import torch
import torch.nn as nn
import torchvision
import numpy as np
import pdb
import wandb


class EfficientNet(nn.Module):
    def __init__(self, num_classes=100) -> None:
        super(EfficientNet, self).__init__()
        self.model = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        )
        self.model.classifier = nn.Identity()
        self.classify = nn.Sequential(
            nn.Dropout(0.4, inplace=True),
            nn.Linear(1280, num_classes),
        )

    def forward(self, inputs, return_features=False):
        emb = self.model(inputs)
        if return_features:
            return emb.view(emb.size(0), -1)
        else:
            return self.classify(emb)
