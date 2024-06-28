
import numpy as np
from torch import nn


class ResNetBlockFacConv(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_feature, out_feature, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_feature)
        self.conv_dw = nn.Conv2d(
            out_feature, out_feature, kernel_size=3, stride=1, padding=1,
            groups=out_feature)
        self.bn2 = nn.BatchNorm2d(out_feature)
        self.conv1x1_2 = nn.Conv2d(
            out_feature, out_feature, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn1(x)
        x = self.relu(x)
        identity = x
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv1x1_2(x)
        x = self.bn3(x)
        x += identity
        x = self.relu(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, features_in, feature_list):
        super().__init__()
        self.patch_emb = nn.Conv2d(
            3, features_in, kernel_size=16, stride=16)

        feature_list = np.stack([np.roll(feature_list, 1), feature_list]).T
        feature_list[0, 0] = features_in
        self.encoder = nn.Sequential(*[
            ResNetBlockFacConv(f1, f2) for f1, f2 in feature_list])

    def forward(self, x):
        x = self.patch_emb(x)
        x = self.encoder(x)
        return x


class ZeyuNet20(nn.Module):
    NAME="zeyu20"
    def __init__(self):
        super().__init__()

        self.feature_in = 768
        self.feature_list = [
          384,
          96, 96,
          48, 48,
          24, 24, 24, 24,
          12, 12, 12, 12,
          6, 6, 6, 6, 6,
          3
        ]
        
        self.enc = ResNetEncoder(self.feature_in, self.feature_list)
        self.mlp = nn.Sequential(
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

    def forward(self, x):
        x = self.enc(x)
        return self.mlp(x)
