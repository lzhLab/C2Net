import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """
    Convolution + BatchNorm + ReLU.
    """
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


class MGUCell(nn.Module):
    """
    Minimal Gated Unit cell.

    Equations:
        f_t = sigmoid(W_f x_t + U_f h_{t-1})
        h_hat_t = tanh(W_h x_t + U_h (f_t * h_{t-1}))
        h_t = (1 - f_t) * h_{t-1} + f_t * h_hat_t
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x_to_f = nn.Linear(input_size, hidden_size, bias=True)
        self.h_to_f = nn.Linear(hidden_size, hidden_size, bias=False)

        self.x_to_h = nn.Linear(input_size, hidden_size, bias=True)
        self.h_to_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t, h_prev):
        f_t = torch.sigmoid(self.x_to_f(x_t) + self.h_to_f(h_prev))
        h_hat = torch.tanh(self.x_to_h(x_t) + self.h_to_h(f_t * h_prev))
        h_t = (1.0 - f_t) * h_prev + f_t * h_hat

        return h_t


class MGUSequence(nn.Module):
    """
    One-direction MGU sequence layer.

    Input:
        x: [B, L, input_size]

    Output:
        outputs: [B, L, hidden_size]
        h_t:     [B, hidden_size]
    """
    def __init__(self, input_size, hidden_size, reverse=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.reverse = reverse
        self.cell = MGUCell(input_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h_t = x.new_zeros(batch_size, self.hidden_size)

        outputs = []
        indices = range(seq_len - 1, -1, -1) if self.reverse else range(seq_len)

        for t in indices:
            h_t = self.cell(x[:, t, :], h_t)
            outputs.append(h_t)

        if self.reverse:
            outputs = outputs[::-1]

        outputs = torch.stack(outputs, dim=1)

        return outputs, h_t


class BiMGU(nn.Module):
    """
    Bidirectional MGU layer.

    Input:
        x: [B, L, input_size]

    Output:
        out: [B, L, hidden_size * 2]
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.forward_mgu = MGUSequence(
            input_size=input_size,
            hidden_size=hidden_size,
            reverse=False
        )

        self.backward_mgu = MGUSequence(
            input_size=input_size,
            hidden_size=hidden_size,
            reverse=True
        )

    def forward(self, x):
        out_f, _ = self.forward_mgu(x)
        out_b, _ = self.backward_mgu(x)

        out = torch.cat([out_f, out_b], dim=-1)

        return out


class AxialContinuityMGU(nn.Module):
    """
    Row-wise and column-wise bidirectional MGU module for continuity estimation.

    Input:
        x: [B, C, H, W]

    Output:
        continuity_map: [B, 1, H, W], values in [0, 1]

    This module avoids invalid unfold operations such as kernel_size=(C, k).
    Instead, it treats rows and columns as sequences:
        row_seq: [B * H, W, hidden]
        col_seq: [B * W, H, hidden]
    """
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.row_mgu = BiMGU(
            input_size=hidden_channels,
            hidden_size=hidden_channels
        )

        self.col_mgu = BiMGU(
            input_size=hidden_channels,
            hidden_size=hidden_channels
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape

        x = self.reduce(x)  # [B, hidden, H, W]

        # Row-wise sequence modeling: each row is a sequence of length W.
        row_seq = x.permute(0, 2, 3, 1).contiguous()
        row_seq = row_seq.view(batch_size * height, width, self.hidden_channels)
        row_out = self.row_mgu(row_seq)
        row_out = row_out.view(batch_size, height, width, self.hidden_channels * 2)
        row_out = row_out.permute(0, 3, 1, 2).contiguous()

        # Column-wise sequence modeling: each column is a sequence of length H.
        col_seq = x.permute(0, 3, 2, 1).contiguous()
        col_seq = col_seq.view(batch_size * width, height, self.hidden_channels)
        col_out = self.col_mgu(col_seq)
        col_out = col_out.view(batch_size, width, height, self.hidden_channels * 2)
        col_out = col_out.permute(0, 3, 2, 1).contiguous()

        continuity_feature = torch.cat([row_out, col_out], dim=1)
        continuity_map = torch.sigmoid(self.fuse(continuity_feature))

        return continuity_map


class CCRBlock(nn.Module):
    """
    Continuity-guided convolutional residual block.

    If continuity_map is provided:
        out = out + out * continuity_map

    The continuity map is broadcast along the channel dimension.
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

    Settings:
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

    Architecture:
        Encoder: CCR blocks
        Continuity module: axial bidirectional MGU on bottleneck feature
        Decoder: atrous convolution blocks

    Encoder channels:
        [base_channels, base_channels*2, base_channels*4, base_channels*8]

    Default:
        base_channels = 32
        channels = [32, 64, 128, 256]
    """
    def __init__(self, in_channels=3, num_classes=1, base_channels=32, mgu_hidden_channels=None):
        super().__init__()

        if mgu_hidden_channels is None:
            mgu_hidden_channels = max(base_channels // 2, 16)

        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8
        ]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.mgu_hidden_channels = mgu_hidden_channels
        self.channels = channels

        self.enc1 = CCRBlock(in_channels, channels[0])
        self.enc2 = CCRBlock(channels[0], channels[1])
        self.enc3 = CCRBlock(channels[1], channels[2])
        self.enc4 = CCRBlock(channels[2], channels[3])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.continuity_mgu = AxialContinuityMGU(
            in_channels=channels[3],
            hidden_channels=mgu_hidden_channels
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
        # First encoder pass: obtain bottleneck feature for continuity estimation.
        x1_raw = self.enc1(x)
        x2_raw = self.enc2(self.pool(x1_raw))
        x3_raw = self.enc3(self.pool(x2_raw))
        x4_raw = self.enc4(self.pool(x3_raw))

        continuity_map = self.continuity_mgu(x4_raw)

        self.last_continuity_map = self._resize_map(
            continuity_map,
            size=x.shape[2:]
        )

        # Guided encoder pass: apply continuity guidance to all encoder stages.
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
        """
        Return the upsampled continuity map from the latest forward pass.

        Shape:
            [B, 1, H, W]
        """
        if self.last_continuity_map is None:
            raise RuntimeError(
                "Continuity map is not available. Run model(x) before calling get_continuity_map()."
            )

        return self.last_continuity_map


class RNN_Model(CCRUNet):
    """
    Compatibility wrapper.

    Supported initialization styles:
        RNN_Model(input_channels=3, output_channels=1)
        RNN_Model(in_channels=3, out_channels=1)
        RNN_Model(input_channels=3, output_channels=1, base_channels=32)
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
        mgu_hidden_channels=None,
        rnn_hidden_channels=None
    ):
        if in_channels is not None:
            input_channels = in_channels

        if out_channels is not None:
            output_channels = out_channels

        if num_classes is not None:
            output_channels = num_classes

        if mgu_hidden_channels is None:
            mgu_hidden_channels = rnn_hidden_channels

        super().__init__(
            in_channels=input_channels,
            num_classes=output_channels,
            base_channels=base_channels,
            mgu_hidden_channels=mgu_hidden_channels
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

