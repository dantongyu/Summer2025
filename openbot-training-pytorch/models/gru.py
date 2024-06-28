
import torch
from torch import nn

from .base import ModelBase
from .common import CONFIG, CropKernel, EdgeKernel


class GRUSeq(ModelBase):
    NAME = "gru"
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.MSELoss()
        self.filter = EdgeKernel(3) if CONFIG["edge_filter"] else nn.Identity()
        self.crop = CropKernel(3, 90) if CONFIG["crop_filter"] else nn.Identity()

        self.hidden_state = 500
        self.backbone = torch.hub.load(
                    'pytorch/vision:v0.10.0', "resnet18")
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_state, 128),
            nn.ReLU(),
            nn.Linear(128, CONFIG["output_features"]*2),
        )

        self.model = nn.GRU(
            input_size=1000,
            hidden_size=self.hidden_state,
            num_layers=2,
            batch_first=True,
        )

    def inference(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        '''
        Inference pass of the model
        Args:
            X_i (torch.Tensor): input image (3, 224, 224)
            h_i (torch.Tensor): hidden state (2, 1000)
        Returns:
            X_{i+1} (torch.Tensor): output image (2)
            h_{i+1} (torch.Tensor): hidden state (2, 1000)
        '''
        x = x.unsqueeze(0)
        x = self.filter(x)
        x = self.crop(x)
        x = self.backbone(x)
        x, h = self.model(x, h)
        x = self.decoder(x)
        return x.squeeze(0), h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model
        Args:
            X (torch.Tensor): input image (batch_size, seq_len, 3, 224, 224)
        Returns:
            X (torch.Tensor): output image (batch_size, seq_len, 2)
        '''
        bs, seqlen, c, h, w = x.shape
        x = x.view(bs * seqlen, c, h, w)
        x = self.filter(x)
        x = self.crop(x)
        x = self.backbone(x)
        x = x.view(bs, seqlen, -1)
        x, _ = self.model(x)
        x = self.decoder(x)
        return x
