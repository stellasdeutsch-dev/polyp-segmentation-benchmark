"""U-Net written from scratch (rung 1 of the ladder, decision D12).

Ronneberger, Fischer and Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation",
MICCAI 2015, arXiv:1505.04597. Four times down, four times up, 64 channels doubling to 1,024 at
the bottom, and the feature map of each encoder level concatenated into the decoder level of the
same size (the skip connections).

Where we follow milesial/Pytorch-UNet instead of the paper (research/02_unet_lineage.md):
  padded 3x3 convolutions   the output has the input's size, so no cropping of skips or masks;
                            the paper's unpadded version turns 572 px into 388 px
  BatchNorm after each conv the paper predates BatchNorm; it makes training from scratch stable
  no bias before BatchNorm  BatchNorm's own shift makes it redundant
  no dropout                the paper puts dropout at the end of the encoder; BatchNorm and
                            augmentation do the regularising here, as in milesial
Kept from the paper: 2x2 max-pooling, 2x2 transposed-convolution upsampling that halves the
channels, a final 1x1 convolution, and He initialisation (normal, std = sqrt(2 / fan_in)).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

WIDTHS = (64, 128, 256, 512, 1024)


class DoubleConv(nn.Sequential):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU. Keeps the spatial size."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class Down(nn.Sequential):
    """Halve the size with 2x2 max-pooling, then a double convolution."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__(nn.MaxPool2d(2), DoubleConv(c_in, c_out))


class Up(nn.Module):
    """Double the size with a 2x2 transposed convolution, concatenate the skip, double convolution."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(c_in, c_out)  # c_in // 2 upsampled + c_in // 2 from the skip

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dy or dx:  # odd sizes on the way down lose a pixel; pad back to the skip's size
            x = F.pad(x, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """(B, in_channels, H, W) images to (B, classes, H, W) logits, for any H and W."""

    def __init__(self, in_channels: int = 3, classes: int = 1, widths: tuple[int, ...] = WIDTHS):
        super().__init__()
        self.inc = DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList(Down(a, b) for a, b in zip(widths[:-1], widths[1:], strict=True))
        rev = widths[::-1]
        self.ups = nn.ModuleList(Up(a, b) for a, b in zip(rev[:-1], rev[1:], strict=True))
        self.head = nn.Conv2d(widths[0], classes, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """He initialisation for every convolution, as in the paper (section 3)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())
        return self.head(x)
