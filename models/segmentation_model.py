"""
Segmentation model architecture for Assignment 03.

This module defines a fully convolutional encoder-decoder network that mirrors
the architecture sketched in the assignment brief. The upcoming data loader
should hand batches to the `forward` method as dictionaries in the following
shape:

    batch = {
        "image": torch.FloatTensor[B, C, H, W],  # RGB image normalized by the loader
        "mask": torch.LongTensor[B, H, W],       # Optional segmentation mask (ground truth)
        ...
    }

Only the `"image"` entry is consumed by the network, but the `"mask"` can be
kept in the batch for loss computation outside the model. Keeping the dict
structure makes it easy to extend the loader later with auxiliary inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn


@dataclass
class SegmentationModelConfig:
    """
    Configuration for the segmentation model.

    All attributes are intentionally explicit and editable so the architecture
    can be adapted quickly once dataset specifics are known.
    """

    in_channels: int = 3  # RGB input image
    num_classes: int = 2  # Foreground / background by default
    encoder_channels: Tuple[int, ...] = (64, 128, 256, 512)
    decoder_channels: Tuple[int, ...] = (512, 256, 128, 64)
    kernel_size: int = 3
    padding: int = 1
    dropout_p: float = 0.1  # Set to 0 to disable dropout


class ConvBlock(nn.Module):
    """Two consecutive Conv-BN-ReLU layers used throughout encoder and decoder."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, padding: int, dropout_p: float):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.insert(-1, nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegmentationModel(nn.Module):
    """
    Fully convolutional encoder-decoder with pooling indices based unpooling.

    The figure in the assignment shows a set of blue encoder stages (downsampling)
    followed by orange decoder stages (upsampling). Each encoder stage stores
    pooling indices that are later used by the decoder to guide `MaxUnpool2d`,
    enabling sharper reconstructions.
    """

    def __init__(self, config: SegmentationModelConfig | None = None):
        super().__init__()
        self.config = config or SegmentationModelConfig()

        self.encoder_blocks = self._build_blocks(
            self.config.in_channels,
            self.config.encoder_channels,
            block_type="encoder",
        )
        self.decoder_blocks = self._build_blocks(
            self.config.encoder_channels[-1],
            self.config.decoder_channels,
            block_type="decoder",
        )

        self.pool_layers = nn.ModuleList(
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) for _ in self.encoder_blocks
        )
        self.unpool_layers = nn.ModuleList(nn.MaxUnpool2d(kernel_size=2, stride=2) for _ in self.decoder_blocks)

        if len(self.encoder_blocks) != len(self.decoder_blocks):
            raise ValueError(
                "Encoder and decoder depth must match to pair pooling indices with unpooling layers. "
                f"Got {len(self.encoder_blocks)} encoder blocks and {len(self.decoder_blocks)} decoder blocks."
            )

        self.classifier = nn.Conv2d(self.config.decoder_channels[-1], self.config.num_classes, kernel_size=1)

    def _build_blocks(self, in_channels: int, channel_sequence: Iterable[int], block_type: str) -> nn.ModuleList:
        blocks: List[nn.Module] = []
        current_in = in_channels
        for out_ch in channel_sequence:
            blocks.append(
                ConvBlock(
                    current_in,
                    out_ch,
                    kernel_size=self.config.kernel_size,
                    padding=self.config.padding,
                    dropout_p=self.config.dropout_p if block_type == "encoder" else 0.0,
                )
            )
            current_in = out_ch
        return nn.ModuleList(blocks)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        batch:
            Dictionary produced by the upcoming data loader. The `"image"` key
            MUST map to the input tensor (B, C, H, W). The `"mask"` entry, if
            present, is ignored here but the training loop can consume it.

        Returns
        -------
        torch.Tensor
            Raw logits of shape (B, num_classes, H, W) ready for segmentation loss.
        """

        # --- Encoder (blue blocks in the assignment figure) ---
        x = batch["image"]  # <-- DATA LOADER FEEDS THE PREPROCESSED IMAGE HERE
        skip_connections: List[Tuple[torch.Size, torch.Tensor]] = []

        for block, pool in zip(self.encoder_blocks, self.pool_layers):
            x = block(x)
            size_before_pool = x.size()
            x, indices = pool(x)
            skip_connections.append((size_before_pool, indices))

        # --- Decoder (orange blocks, using stored pooling indices) ---
        for block, unpool in zip(self.decoder_blocks, self.unpool_layers):
            size, indices = skip_connections.pop()
            x = unpool(x, indices, output_size=size)
            x = block(x)

        # --- Classifier head (yellow block leading to segmentation output) ---
        logits = self.classifier(x)
        return logits


def create_model(overrides: Dict[str, object] | None = None) -> SegmentationModel:
    """
    Factory helper to build the model with partial config overrides.

    Example
    -------
    >>> model = create_model({"num_classes": 2})
    >>> dummy_batch = {"image": torch.randn(2, 3, 256, 256)}
    >>> logits = model(dummy_batch)
    >>> print(logits.shape)
    torch.Size([2, 2, 256, 256])
    """

    overrides = overrides or {}
    config_dict = {**SegmentationModelConfig().__dict__, **overrides}
    config = SegmentationModelConfig(**config_dict)
    return SegmentationModel(config=config)


__all__ = ["SegmentationModelConfig", "SegmentationModel", "create_model"]
