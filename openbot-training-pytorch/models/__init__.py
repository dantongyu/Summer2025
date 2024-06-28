'''
Name: models.py
Description: This program is used to define the neural networks using pytorch
            It also contains the ModelHub class which is used to load and save
            models. Helper functions such as get_model and get_model_openbot
            are for ease of loading models in seperate scripts
Date: 2023-08-25
Date Modified: 2023-08-25
'''
from torch import nn

from .cnn import CNN
from .gru import GRUSeq
from .hub import ModelHubBase
from .mixer import ZeyuNet20
from .vit import ViT, ViT0
from .yolo import Yolo
from .efficientnet import EfficientNet
from .fastvit import FastViT, FastViT1


models = {
    "cnn": CNN,
    "vit": ViT,
    "vit0": ViT0,
    "yolo": Yolo,
    "zeyu20": ZeyuNet20,
    "gru": GRUSeq,
    "efficientnet": EfficientNet,
    "fastvit": FastViT,
    "fastvit1": FastViT1,
}


def get_model(model_name):
    '''
    Get model by name from either the models dict or from hub
    Args:
        model_name (str): name of model
    Returns:
        model (nn.Module): model loaded from hub
    '''
    if model_name in models:
        return models[model_name]
    else:
        return ModelHubBase.load_model(model_name)


def get_model_permuteChannels(model_name : str) -> nn.Module:
    model_cls = get_model(model_name)
    class Model(model_cls):
        def forward(self, x, h):
            # Hotfix for openbot input and output format
            # [B,C,H,W] 
            x = x.squeeze(0)
            # [C,H,W]  ~ BGR
            x = x[[2, 1, 0], ...]
            # [C,H,W]  ~ RGB
            x, h = super().inference(x, h)
            return x, h

    return Model


def get_model_openbot(model_name : str) -> nn.Module:
    '''
    Get model by name from either the models dict or from hub
    Then wrap it in a class that has outputs compatible with openbot
    Args:
        model_name (str): name of model
    Returns:
        model (nn.Module): model loaded from hub
    '''
    model_cls = get_model(model_name)
    class Model(model_cls):
        def forward(self, x):
            # Hotfix for openbot input and output format
            x = x.permute(0, 3, 1, 2)
            x = super().forward(x)
            x = x[..., [1, 0]]
            return x

    return Model
