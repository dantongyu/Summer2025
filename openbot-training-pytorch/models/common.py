import config
import torch
from config import CONFIG
from torch import nn


def load_config(file):
    global CONFIG
    CONFIG = config.set_config(file)


class AvgPool11(nn.Module):
    '''
    AvgPool11 Class - used to define a simple average pooling layer
    Args:
        None
    Methods:
        forward(X): forward pass of the model
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model
        Args:
            X (torch.Tensor): input image
        Returns:
            X (torch.Tensor): output image
        '''
        return x.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)


class CropKernel(nn.Module):
    def __init__(self, channels, horizon):
        super().__init__()
        self.crop_matrix = nn.Parameter(
            torch.cat((torch.zeros(channels,224-horizon,224), torch.ones(channels,horizon,224)),1),
            requires_grad=False,
        )

    def forward(self, X):
        Y = X*self.crop_matrix
        return Y


class EdgeKernel(nn.Module):
    '''
    EdgeKernel class - used to define a convolutional layer that filters edges from images

    Args: 
        channels (int): number of channels in input image
        kernel (int): size of kernel to use for convolution

    Methods:
        forward(X): forward pass of the model
    '''
    def __init__(self, channels : int = 3, kernel : int = 3) -> None:
        super().__init__()
        self.Gx = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel, padding=1)
        self.Gx.weight.data = torch.zeros(*self.Gx.weight.data.shape)
        self.Gy = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel, padding=1)
        self.Gy.weight.data = torch.zeros(*self.Gy.weight.data.shape)
        Kx = torch.as_tensor([[1,0,-1],[2,0,-2],[1,0,-1]])
        for i in range(channels):
            self.Gx.weight.data[i, i] = Kx
            self.Gy.weight.data[i, i] = Kx.T

        self.Gy.weight.requires_grad = False
        self.Gx.weight.requires_grad = False

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model
        Args:
            X (torch.Tensor): input image
        Returns:
            X (torch.Tensor): output image
        '''
        X = torch.sqrt(self.Gx(X)**2 + self.Gy(X)**2)
        X = X / X.amax(dim=(2, 3), keepdim=True)
        return X
