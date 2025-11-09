# models/unet.py
import math
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv-BN-ReLU) x 2"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    """
    U-Net (encoder-decoder with skip connections)
    - depth: number of down/upsampling stages (default 4)
    - base_c: channels in the first stage (default 64)
    Output is logits (no sigmoid). For binary segmentation, use BCEWithLogits or sigmoid at eval.
    """
    def __init__(self, in_channels=3, out_channels=1, base_c=64, depth=4):
        super().__init__()
        assert depth >= 1
        chs = [base_c * (2 ** i) for i in range(depth)]

        # Down path
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for c in chs:
            self.downs.append(DoubleConv(prev, c))
            self.pools.append(nn.MaxPool2d(2))
            prev = c

        # Bottleneck
        self.bottleneck = DoubleConv(prev, prev * 2)

        # Up path
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        up_c = prev * 2
        for c in reversed(chs):
            self.ups.append(nn.ConvTranspose2d(up_c, c, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(up_c, c))  # concat(skip, up) => channels up_c
            up_c = c

        self.final = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["image"]

        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, up_conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            # If odd shapes arise from resizing, center-crop skip to match x
            if x.shape[-2:] != skip.shape[-2:]:
                dh = skip.shape[-2] - x.shape[-2]
                dw = skip.shape[-1] - x.shape[-1]
                skip = skip[..., dh//2:skip.shape[-2]-(dh-dh//2),
                            dw//2:skip.shape[-1]-(dw-dw//2)]
            x = torch.cat([skip, x], dim=1)
            x = up_conv(x)

        return self.final(x)  # logits


def create_unet(
    num_classes: int = 2,
    in_channels: int = 3,
    base_channels: int = 64,
    depth: int = 4,
) -> UNet:
    """
    Helper to instantiate UNet with configurable number of classes.
    """
    return UNet(in_channels=in_channels, out_channels=num_classes, base_c=base_channels, depth=depth)


__all__ = ["UNet", "create_unet"]
