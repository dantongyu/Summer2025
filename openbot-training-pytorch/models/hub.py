
import config
import torch
from config import CONFIG
from torch import nn

from .base import ModelBase
from .common import CropKernel, EdgeKernel


def load_config(file):
    global CONFIG
    CONFIG = config.set_config(file)


class ModelHubBase(ModelBase):
    '''
    ModelHubBase Class - generic class for loading and using hub models, not to be used directly
                        but instead with get_model function

    Args:
        edge_filter (bool): whether to filter edges from input image
        old_model (bool): whether to use old model (for backward compatibility)

    Methods:
        forward(X): forward pass of the model
        load_model(model_name): load model from hub by name
    '''
    NAME = None
    def __init__(self) -> None:
        super().__init__()
        self.filter = EdgeKernel(3) if CONFIG["edge_filter"] else nn.Identity()
        self.crop = CropKernel(3,108) if CONFIG["crop_filter"] else nn.Identity()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', self.NAME)
        if hasattr(self.model, "features"):
            # This fix some of the model in the following structure:
            # features -> adaptive_pooling -> classifier
            # the adaptive_pooling is not supported by onnx, so we need
            # define a new output layer
            self.model = self.model.features
            self.output = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                nn.Dropout(p=0.2),
                nn.LazyLinear(128),
                nn.ReLU(),
                nn.Linear(128, CONFIG["output_features"]*2),
            )
        else:
            self.output = nn.LazyLinear(CONFIG["output_features"]*2)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model
        Args:
            X (torch.Tensor): input image
        Returns:
            X (torch.Tensor): output image
        '''
        #first convert image from BGR to RGB
        x = self.filter(x)
        x = self.crop(x)
        x = self.model(x)
        #commented for conversion
        #if self.NAME == "googlenet":  # Fix ouput of googlenet
            #x = x[0]

        x = self.output(x)
        return x

    @classmethod
    def load_model(cls, model_name : str) -> nn.Module:
        '''
        Load model from hub by name
        Args:
            model_name (str): name of model
        Returns:
            model (ModelHubBase): model loaded from hub
        '''
        # This is a hack for compatibility
        # This method create a new subclass of ModelHubBase
        # and set the NAME attribute to the model_name
        cls_instance = type(model_name, (cls,), {})
        cls_instance.NAME = model_name
        return cls_instance
