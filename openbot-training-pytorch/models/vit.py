import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

from .base import ModelBase


class ViT(ModelBase):
    NAME = "vit"

    def __init__(self, weight=True):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if weight else None)
        self.model.heads = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 2)
        )

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.model._process_input(x)

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.model.heads(x)

        return x


class ViT0(ViT):
    NAME = "vit0"

    def __init__(self):
        super().__init__(weight=False)
