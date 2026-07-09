import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class CCRBlock(nn.Module):
    """
    Continuity-guided convolutional residual unit.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            dilation=1
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + identity
        out = self.relu(out)

        return out


class AtrousDecoderBlock(nn.Module):
    """
    Decoder block with 3x3 atrous convolution.

    kernel_size = 3
    padding = 2
    dilation = 2
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv1 = ConvBNReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=2,
            dilation=2
        )

        self.conv2 = ConvBNReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=2,
            dilation=2
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class CCRUNet(nn.Module):
    """
    U-shaped network with CCR encoder and atrous convolution decoder.

    Encoder stages:
    32, 64, 128, 256

    Decoder:
    3x3 atrous convolution with padding=2 and dilation=2.
    """
    def __init__(self, in_channels=3, num_classes=1, base_channels=32):
        super().__init__()

        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8
        ]

        self.enc1 = CCRBlock(in_channels, channels[0])
        self.enc2 = CCRBlock(channels[0], channels[1])
        self.enc3 = CCRBlock(channels[1], channels[2])
        self.enc4 = CCRBlock(channels[2], channels[3])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dec3 = AtrousDecoderBlock(
            in_channels=channels[3],
            skip_channels=channels[2],
            out_channels=channels[2]
        )

        self.dec2 = AtrousDecoderBlock(
            in_channels=channels[2],
            skip_channels=channels[1],
            out_channels=channels[1]
        )

        self.dec1 = AtrousDecoderBlock(
            in_channels=channels[1],
            skip_channels=channels[0],
            out_channels=channels[0]
        )

        self.out_conv = nn.Conv2d(
            channels[0],
            num_classes,
            kernel_size=1
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        d3 = self.dec3(x4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)

        out = self.out_conv(d1)

        return out


class RNN_Model(CCRUNet):
    """
    Compatibility wrapper for the original training code.

    The original code may call:
        RNN_Model(input_channels, output_channels)

    This wrapper maps it to:
        CCRUNet(in_channels=input_channels, num_classes=output_channels)
    """
    def __init__(self, input_channels=3, output_channels=1, base_channels=32):
        super().__init__(
            in_channels=input_channels,
            num_classes=output_channels,
            base_channels=base_channels
        )


if __name__ == "__main__":
    model = RNN_Model(
        input_channels=3,
        output_channels=1
    )

    x = torch.randn(1, 3, 512, 512)
    y = model(x)

    print(model)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
