from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def direction_metric(pred : torch.Tensor, act : torch.Tensor) -> torch.Tensor:
    '''
    Direction metric - how well the model predicts the direction of the car
    Args:
        pred (torch.Tensor): predicted values
        act (torch.Tensor): actual values
    Returns:
        torch.Tensor: direction metric
    '''
    angle_true = act[...,0]
    angle_pred = pred[...,0]
    turns = torch.abs(angle_true) > 0.1
    logits = torch.sign(angle_pred[turns]) == torch.sign(angle_true[turns])
    dir_metric, num = torch.sum(logits.float()), len(logits)
    return dir_metric.item() / num if num > 0 else np.nan


def angle_metric(pred : torch.Tensor, act : torch.Tensor) -> torch.Tensor:
    '''
    Angle metric - how well the model predicts the angle of the car
    Args:
        pred (torch.Tensor): predicted values
        act (torch.Tensor): actual values
    Returns:
        torch.Tensor: angle metric
    '''
    angle_true = act[...,0]
    angle_pred = pred[...,0]
    logits = torch.abs(angle_true - angle_pred) < 0.02
    return torch.mean(logits.float()).item()


class ModelBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward_loss(self, D) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(D, tuple):
            # Supervised
            x, y = D
        else:
            # Autoencoder
            x, y = D, D

        z = self.forward(x)
        return self.loss_fn(z, y), z

    def encode_output(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def decode_output(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_output(self.forward(x.unsqueeze(0))).squeeze(0)

    def loss_fn(self, pred: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, self.encode_output(Y))

    def metrics(self, pred: torch.Tensor, D) -> dict:
        Y_pred = self.decode_output(pred)
        X, Y = D
        return {
            "angle_err": 1 - angle_metric(Y_pred, Y),
            "direction_err": 1 - direction_metric(Y_pred, Y),
        }
