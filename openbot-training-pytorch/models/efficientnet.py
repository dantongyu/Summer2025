import config
import torch
from config import CONFIG
from torch import nn

from .base import ModelBase
from .common import CropKernel, EdgeKernel


def load_config(file):
    global CONFIG
    CONFIG = config.set_config(file)


class EfficientNet(ModelBase):
    NAME = "efficientnet"
    def __init__(self, pretrained=True):
        super().__init__()

        self.filter = EdgeKernel(3) if CONFIG["edge_filter"] else nn.Identity()
        self.crop = CropKernel(3,108) if CONFIG["crop_filter"] else nn.Identity()

        self.encoder = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', "nvidia_efficientnet_b0",
            pretrained=pretrained)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        x = self.filter(x)
        x = self.crop(x)

        x = self.encoder.stem(x)
        x = self.encoder.layers(x)
        x = self.encoder.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return self.decoder(x)
