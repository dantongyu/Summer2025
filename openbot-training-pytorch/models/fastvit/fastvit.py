import os
from typing import List, Tuple

import torch
from torch import nn

from .blocks import (FastVITAttentionBlock, FastVITRepMixerBlock, PatchEmbed,
                     convolutional_stem, convolutional_stem_inv)

class FastViT(nn.Module):
    """
    This class implements `FastViT architecture <https://arxiv.org/pdf/2303.14189.pdf>`_
    """
    # Prefixed with:
    #   "S": models with smaller embedding dimensions, i.e. [64,128,256, 512]
    #   "SA": contain Self Attention layers
    #   "M": models with bigger embedding dimensions,i.e. [76,152,304,608]
    #   "T": Models with MLP expansion ratio less than 4.
    # The number in the notation denotes total number of FastViTblocks.

    def __init__(
        self,
        layers,
        token_mixers: Tuple[str],
        embed_dims=None,
        mlp_ratios=None,
        downsamples=None,
        repmixer_kernel_size=3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU,
        pos_embs=None,
        down_patch_size=7,
        down_stride=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        fork_feat=True,
        inference_mode=False,
    ) -> None:

        super().__init__()

        self.fork_feat = fork_feat

        if pos_embs is None:
            pos_embs = [None] * len(layers)

        # Convolutional stem
        self.patch_embed = convolutional_stem(3, embed_dims[0], inference_mode=False)
        self.unpatch_embed = convolutional_stem_inv(embed_dims[0], 3, inference_mode=False)
        # Build the main stages of the network architecture
        network = []
        for i in range(len(layers)):
            # Add position embeddings if requested
            if pos_embs[i] is not None:
                network.append(
                    pos_embs[i](
                        embed_dims[i], embed_dims[i], inference_mode=inference_mode
                    )
                )
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                token_mixer_type=token_mixers[i],
                kernel_size=repmixer_kernel_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break

            # Patch merging/downsampling between stages.
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        in_channels=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                        inference_mode=inference_mode,
                    )
                )

        self.network = nn.ModuleList(network)

        decoder_network = []

        for i in range(len(layers)):
            i_inverse = -1-i
            stage = basic_blocks(
                embed_dims[i_inverse],
                i,
                layers,
                token_mixer_type=token_mixers[i_inverse],
                kernel_size=repmixer_kernel_size,
                mlp_ratio=mlp_ratios[i_inverse],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
            )
            decoder_network.append(stage)

            if pos_embs[i_inverse] is not None:
                decoder_network.append(
                    pos_embs[i_inverse](
                        embed_dims[i_inverse], embed_dims[i_inverse], inference_mode=inference_mode
                    )
                )
            if i >= len(layers) - 1:
                break

            # # Patch merging/downsampling between stages.
            if downsamples[i_inverse] or embed_dims[i_inverse] != embed_dims[i_inverse-1]:
                decoder_network.append(nn.ConvTranspose2d(embed_dims[i_inverse], embed_dims[i_inverse-1], 2, stride=2))

        self.decoder_network = decoder_network

        # For segmentation and detection, extract intermediate output
        # add a norm layer for each output
        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            if i_emb == 0 and os.environ.get("FORK_LAST3", None):
                """For RetinaNet, `start_level=1`. The first norm layer will not used.
                cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                """
                layer = nn.Identity()
            else:
                layer = norm_layer(embed_dims[i_emb])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        # input embedding
        x = self.patch_embed(x)

        # through backbone
        for idx, block in enumerate(self.network):
            x = block(x)

        x_out = self.norm6(x)

        # output the features of four stages for dense prediction
        return x_out

    def forward_decoder(self, x: torch.Tensor) -> torch.Tensor:
        for _, block in enumerate(self.decoder_network):
            x = block(x)

        x = self.unpatch_embed(x)
        return x


def basic_blocks(
    dim: int,
    block_index: int,
    num_blocks: List[int],
    token_mixer_type: str,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    act_layer: nn.Module = nn.GELU,
    norm_layer: nn.Module = nn.BatchNorm2d,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
    inference_mode=False,
) -> nn.Sequential:
    """Build FastViT blocks within a stage.

    Args:
        dim: Number of embedding dimensions.
        block_index: block index.
        num_blocks: List containing number of blocks per stage.
        token_mixer_type: Token mixer type.
        kernel_size: Kernel size for repmixer.
        mlp_ratio: MLP expansion ratio.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        drop_rate: Dropout rate.
        drop_path_rate: Drop path rate.
        use_layer_scale: Flag to turn on layer scale regularization.
        layer_scale_init_value: Layer scale value at initialization.
        inference_mode: Flag to instantiate block in inference mode.

    Returns:
        nn.Sequential object of all the blocks within the stage.
    """
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = (
            drop_path_rate
            * (block_idx + sum(num_blocks[:block_index]))
            / (sum(num_blocks) - 1)
        )
        # print(token_mixer_type)
        if token_mixer_type == "repmixer":
            blocks.append(
                FastVITRepMixerBlock(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode,
                )
            )
        elif token_mixer_type == "attention":
            blocks.append(
                FastVITAttentionBlock(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        else:
            raise ValueError(
                "Token mixer type: {} not supported".format(token_mixer_type)
            )
    blocks = nn.Sequential(*blocks)

    return blocks
