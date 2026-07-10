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


class AxialContinuityRNN(nn.Module):
    """
    Row-wise and column-wise bidirectional GRU module for continuity estimation.

    Input:
        x: [B, C, H, W]

    Output:
        continuity_map: [B, 1, H, W], values in [0, 1]

    This avoids using unfold with kernel_size=(C, k), which is invalid when
    C is larger than the spatial size.
    """
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.row_rnn = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.col_rnn = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        b, _, h, w = x.shape

        x = self.reduce(x)  # [B, hidden, H, W]

        # Row-wise sequence modeling: each row is a sequence of length W.
        row_seq = x.permute(0, 2, 3, 1).contiguous()
        row_seq = row_seq.view(b * h, w, self.hidden_channels)
        row_out, _ = self.row_rnn(row_seq)
        row_out = row_out.view(b, h, w, self.hidden_channels * 2)
        row_out = row_out.permute(0, 3, 1, 2).contiguous()

        # Column-wise sequence modeling: each column is a sequence of length H.
        col_seq = x.permute(0, 3, 2, 1).contiguous()
        col_seq = col_seq.view(b * w, h, self.hidden_channels)
        col_out, _ = self.col_rnn(col_seq)
        col_out = col_out.view(b, w, h, self.hidden_channels * 2)
        col_out = col_out.permute(0, 3, 2, 1).contiguous()

        continuity_feature = torch.cat([row_out, col_out], dim=1)
        continuity_map = torch.sigmoid(self.fuse(continuity_feature))

        return continuity_map


class CCRBlock(nn.Module):
    """
    Continuity-guided convolutional residual block.

    If continuity_map is provided, the block performs:
        out = out + out * continuity_map

    This enhances continuous vessel-like regions.
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

    def forward(self, x, continuity_map=None):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity

        if continuity_map is not None:
            if continuity_map.shape[2:] != out.shape[2:]:
                continuity_map = F.interpolate(
                    continuity_map,
                    size=out.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
            out = out + out * continuity_map

        out = self.relu(out)

        return out


class AtrousDecoderBlock(nn.Module):
    """
    Decoder block with 3x3 atrous convolution.

    Atrous setting:
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
    C2Net / CCRUNet.

    U-shaped network with:
        Encoder: CCR blocks
        Continuity module: axial bidirectional GRU on bottleneck feature
        Decoder: atrous convolution blocks

    Encoder channels:
        32, 64, 128, 256 when base_channels=32
    """
    def __init__(self, in_channels=3, num_classes=1, base_channels=32, rnn_hidden_channels=None):
        super().__init__()

        if rnn_hidden_channels is None:
            rnn_hidden_channels = base_channels

        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8
        ]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.channels = channels

        self.enc1 = CCRBlock(in_channels, channels[0])
        self.enc2 = CCRBlock(channels[0], channels[1])
        self.enc3 = CCRBlock(channels[1], channels[2])
        self.enc4 = CCRBlock(channels[2], channels[3])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.continuity_rnn = AxialContinuityRNN(
            in_channels=channels[3],
            hidden_channels=rnn_hidden_channels
        )

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

        self.last_continuity_map = None

    @staticmethod
    def _resize_map(x, size):
        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False
        )

    def forward(self, x):
        # First encoder pass obtains bottleneck features for continuity estimation.
        x1_raw = self.enc1(x)
        x2_raw = self.enc2(self.pool(x1_raw))
        x3_raw = self.enc3(self.pool(x2_raw))
        x4_raw = self.enc4(self.pool(x3_raw))

        continuity_map = self.continuity_rnn(x4_raw)
        self.last_continuity_map = self._resize_map(
            continuity_map,
            size=x.shape[2:]
        )

        # Guided encoder pass.
        c1 = self._resize_map(continuity_map, x1_raw.shape[2:])
        x1 = self.enc1(x, c1)

        c2 = self._resize_map(continuity_map, x2_raw.shape[2:])
        x2 = self.enc2(self.pool(x1), c2)

        c3 = self._resize_map(continuity_map, x3_raw.shape[2:])
        x3 = self.enc3(self.pool(x2), c3)

        x4 = self.enc4(self.pool(x3), continuity_map)

        d3 = self.dec3(x4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)

        out = self.out_conv(d1)

        return out

    def get_continuity_map(self):
        if self.last_continuity_map is None:
            raise RuntimeError(
                "Continuity map is not available. Run model(x) before calling get_continuity_map()."
            )

        return self.last_continuity_map


class RNN_Model(CCRUNet):
    """
    Compatibility wrapper.

    Supports both styles:
        RNN_Model(input_channels=3, output_channels=1)
        RNN_Model(in_channels=3, out_channels=1)
        RNN_Model(3, 1)
    """
    def __init__(
        self,
        input_channels=3,
        output_channels=1,
        base_channels=32,
        in_channels=None,
        out_channels=None,
        num_classes=None,
        rnn_hidden_channels=None
    ):
        if in_channels is not None:
            input_channels = in_channels

        if out_channels is not None:
            output_channels = out_channels

        if num_classes is not None:
            output_channels = num_classes

        super().__init__(
            in_channels=input_channels,
            num_classes=output_channels,
            base_channels=base_channels,
            rnn_hidden_channels=rnn_hidden_channels
        )


if __name__ == "__main__":
    model = RNN_Model(
        input_channels=3,
        output_channels=1,
        base_channels=32
    )

    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    c = model.get_continuity_map()

    print(model)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Continuity map shape:", c.shape)
    print("Continuity map range:", float(c.min()), float(c.max()))
