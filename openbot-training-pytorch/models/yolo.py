
from torch import nn


class Yolo(nn.Module):
    NAME = "yolo"
    def __init__(self, in_channels=3, edge_filter=True):
        super().__init__()
        #load yolo
        from ultralytics import YOLO
        self.yolo = YOLO("yolo_pyt.pt").model
        self.yolo.model = nn.Sequential(*list(self.yolo.model.children())[:-1])
        for i, param in enumerate(self.yolo.named_parameters()):
            #if i < 135:
            param[1].requires_grad = False

        #make mlp at bottom
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.LazyLinear(128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.LazyLinear(32),
            nn.ELU(),
            nn.LazyLinear(2),
        )

    def forward(self, image):
        x = self.yolo(image)
        y = self.mlp(x)
        return y
