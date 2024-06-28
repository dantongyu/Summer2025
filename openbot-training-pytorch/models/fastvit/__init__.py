from functools import partial
from typing import Tuple

from torch import Tensor, nn

from ..base import ModelBase
from .blocks import RepCPE
from .fastvit import FastViT as _FastViT


class FastViT(ModelBase):
    NAME = "fastvit"
    def __init__(
            self, layers=[2, 2, 4, 2],
            token_mixers: Tuple[str] = ["repmixer", "repmixer", "repmixer", "attention"],
            embed_dims=[48, 96, 192, 384], mlp_ratios=[3, 3, 3, 3],
            downsamples=[True, True, True, True],
            repmixer_kernel_size=3,
            norm_layer: nn.Module = nn.BatchNorm2d, act_layer: nn.Module = nn.GELU,
            pos_embs=[None, None, None, None], down_patch_size=7, down_stride=2, drop_rate=0,
            drop_path_rate=0, use_layer_scale=True,
            layer_scale_init_value=0.00001, fork_feat=True,
            inference_mode=False) -> None:

        super().__init__()

        if pos_embs is not None:
            _pos_embs = []
            for pos_emb in pos_embs:
                if pos_emb is None:
                    _pos_embs.append(None)
                elif pos_emb == "repcpe":
                    _pos_embs.append(partial(RepCPE, spatial_shape=(7, 7)))
                else:
                    raise ValueError(
                        f"positional embedding {pos_emb} is not supported")

        self.encoder = _FastViT(
            layers, token_mixers, embed_dims, mlp_ratios, downsamples,
            repmixer_kernel_size, norm_layer, act_layer, _pos_embs,
            down_patch_size, down_stride, drop_rate, drop_path_rate,
            use_layer_scale, layer_scale_init_value, fork_feat, inference_mode)

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

    def forward(self, x: Tensor) -> Tensor:
        out = self.encoder.forward_encoder(x)
        return self.decoder(out)


class FastViT1(ModelBase):
    NAME = "fastvit1"
    def __init__(
            self, layers=[2, 2, 4, 2],
            token_mixers: Tuple[str] = ["repmixer", "repmixer", "repmixer", "attention"],
            embed_dims=[48, 96, 192, 384], mlp_ratios=[3, 3, 3, 3],
            downsamples=[True, True, True, True],
            repmixer_kernel_size=3,
            norm_layer: nn.Module = nn.BatchNorm2d, act_layer: nn.Module = nn.GELU,
            pos_embs=[None, None, None, "repcpe"], down_patch_size=7, down_stride=2, drop_rate=0,
            drop_path_rate=0, use_layer_scale=True,
            layer_scale_init_value=0.00001, fork_feat=True,
            inference_mode=False) -> None:

        super().__init__()

        if pos_embs is not None:
            _pos_embs = []
            for pos_emb in pos_embs:
                if pos_emb is None:
                    _pos_embs.append(None)
                elif pos_emb == "repcpe":
                    _pos_embs.append(partial(RepCPE, spatial_shape=(7, 7)))
                else:
                    raise ValueError(
                        f"positional embedding {pos_emb} is not supported")

        self.encoder = _FastViT(
            layers, token_mixers, embed_dims, mlp_ratios, downsamples,
            repmixer_kernel_size, norm_layer, act_layer, _pos_embs,
            down_patch_size, down_stride, drop_rate, drop_path_rate,
            use_layer_scale, layer_scale_init_value, fork_feat, inference_mode)

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

    def forward(self, x: Tensor) -> Tensor:
        out = self.encoder.forward_encoder(x)
        return self.decoder(out)
