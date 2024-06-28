
import torch
from torch import nn

from .common import EdgeKernel
from .base import ModelBase


class CNN(ModelBase):
    ''' 
    CNN class - used to define a simple convolutional neural network

    Args:
        in_channels (int): number of channels in input image
        edge_filter (bool): whether to filter edges from input image
    
    Methods:
        forward(X): forward pass of the model
    '''
    NAME = "cnn"

    def __init__(self, in_channels : int = 3, edge_filter : bool = True) -> None:
        super().__init__()

        self.filter = EdgeKernel(in_channels) if edge_filter else nn.Identity()

        self.image_module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=2, padding=0),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=2, padding=0),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.LazyConv2d(out_channels=96, kernel_size=3, stride=2, padding=0),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.LazyConv2d(out_channels=128, kernel_size=3, stride=2, padding=0),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.LazyConv2d(out_channels=256, kernel_size=3, stride=2, padding=0),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.LazyLinear(64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.LazyLinear(16),
            nn.ELU(),
            nn.LazyLinear(2),
        )

    def forward(self, image : torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of model
        Args:
            image (torch.Tensor): input image
        Returns:
            output (torch.Tensor): output of model
        '''
        image = self.filter(image)
        output = self.image_module(image)
        return output
