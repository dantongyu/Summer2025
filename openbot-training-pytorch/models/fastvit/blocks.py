from typing import Optional, Tuple, Union

import numpy as np
import torch
from timm.models.layers import DropPath, trunc_normal_
from torch import nn

from .modules.mobileone import MobileOneBlock
from .modules.replknet import ReparamLargeKernelConv

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MHSA(nn.Module):
    """Multi-headed Self Attention module.

    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Build MHSA module that can handle 3D or 4D input tensors.

        Args:
            dim: Number of embedding dimensions.
            head_dim: Number of hidden dimensions per head. Default: ``32``
            qkv_bias: Use bias or not. Default: ``False``
            attn_drop: Dropout rate for attention tensor.
            proj_drop: Dropout rate for projection tensor.
        """
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        B, C, H, W = shape
        N = H * W
        if len(shape) == 4:
            x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class FastVITRepMixerBlock(nn.Module):
    """Implementation of Metaformer block with RepMixer as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ):
        """Build RepMixer Block.

        Args:
            dim: Number of embedding dimensions.
            kernel_size: Kernel size for repmixer. Default: 3
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """

        super().__init__()

        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop Path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale.to(x.device) * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x


class FastVITAttentionBlock(nn.Module):
    """Implementation of metaformer block with MHSA as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ):
        """Build Attention Block.

        Args:
            dim: Number of embedding dimensions.
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            norm_layer: Normalization layer. Default: ``nn.BatchNorm2d``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        """

        super().__init__()

        self.norm = norm_layer(dim)
        self.token_mixer = MHSA(dim=dim)

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x


class RepCPE(nn.Module):
    """Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_

    In our implementation, we can reparameterize this module to eliminate a skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
        inference_mode=False,
    ) -> None:
        """Build reparameterizable conditional positional encoding

        Args:
            in_channels: Number of input channels.
            embed_dim: Number of embedding dimensions. Default: 768
            spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, Tuple), (
            f'"spatial_shape" must by a sequence or int, '
            f"get {type(spatial_shape)} instead."
        )
        assert len(spatial_shape) == 2, (
            f'Length of "spatial_shape" should be 2, '
            f"got {len(spatial_shape)} instead."
        )

        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias=True,
            )
        else:
            self.pe = nn.Conv2d(
                in_channels,
                embed_dim,
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                bias=True,
                groups=embed_dim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            x = self.pe(x) + x
            return x

    def reparameterize(self) -> None:
        # Build equivalent Id tensor
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros(
            (
                self.in_channels,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ),
            dtype=self.pe.weight.dtype,
            device=self.pe.weight.device,
        )
        for i in range(self.in_channels):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        # Reparameterize Id tensor and conv
        w_final = id_tensor + self.pe.weight
        b_final = self.pe.bias

        # Introduce reparam conv
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.embed_dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        for para in self.parameters():
            para.detach_()
        self.__delattr__("pe")


class RepMixer(nn.Module):
    """Reparameterizable token mixer.

    For more details, please refer to our paper:
    `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
        self,
        dim,
        kernel_size=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode: bool = False,
    ):
        """Build RepMixer Module.

        Args:
            dim: Input feature map dimension. :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, H, W)`.
            kernel_size: Kernel size for spatial mixing. Default: 3
            use_layer_scale: If True, learnable layer scale is used. Default: ``True``
            layer_scale_init_value: Initial value for layer scale. Default: 1e-5
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(
                    layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale.to(x.device) * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x

    def reparameterize(self) -> None:
        """Reparameterize mixer and norm into a single
        convolutional layer for efficient inference.
        """
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if self.use_layer_scale:
            w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = (
                self.mixer.id_tensor
                + self.mixer.reparam_conv.weight
                - self.norm.reparam_conv.weight
            )
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for para in self.parameters():
            para.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")


class ConvFFN(nn.Module):
    """Convolutional FFN Module."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Build convolutional FFN module.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_channels: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv.to(x.device)(x)
        x = self.fc1.to(x.device)(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2.to(x.device)(x)
        x = self.drop(x)
        return x


def convolutional_stem(
    in_channels: int, out_channels: int, inference_mode: bool = False
) -> nn.Sequential:
    """Build convolutional stem with MobileOne blocks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        inference_mode: Flag to instantiate model in inference mode. Default: ``False``

    Returns:
        nn.Sequential object with stem elements.
    """
    return nn.Sequential(
        MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
    )


def convolutional_stem_inv(
    in_channels: int, out_channels: int, inference_mode: bool = False
) -> nn.Sequential:
    """Build convolutional stem with MobileOne blocks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        inference_mode: Flag to instantiate model in inference mode. Default: ``False``

    Returns:
        nn.Sequential object with stem elements.
    """
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        ),
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels,
        ),
        nn.ZeroPad2d(padding=(0, 1, 0, 1)),
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
        ),
        nn.ZeroPad2d(padding=(0, 1, 0, 1)),
        # nn.ConvTranspose2d(
        #     in_channels=in_channels, 
        #     out_channels=in_channels, 
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     ),
        
        # nn.ConvTranspose2d(
        #     in_channels=in_channels, 
        #     out_channels=out_channels, 
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     groups=1,)
    )


class PatchEmbed(nn.Module):
    """Convolutional patch embedding layer."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        inference_mode: bool = False,
    ) -> None:
        """Build patch embedding layer.

        Args:
            patch_size: Patch size for embedding computation.
            stride: Stride for convolutional embedding layer.
            in_channels: Number of channels of input tensor.
            embed_dim: Number of embedding dimensions.
            inference_mode: Flag to instantiate model in inference mode. Default: ``False``
        """
        super().__init__()
        block = list()
        block.append(
            ReparamLargeKernelConv(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                groups=in_channels,
                small_kernel=3,
                inference_mode=inference_mode,
            )
        )
        block.append(
            MobileOneBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            )
        )
        self.proj = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x
