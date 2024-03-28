import torch
from torch import nn, einsum
import torchvision
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


class DilationDeformConv(nn.Sequential):
    def __init__(self, in_channels, kernel_size=(3, 3), pad_d=1):
        super(DilationDeformConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            DeformConv(in_channels, kernel_size=kernel_size, padding=pad_d, groups=in_channels, dilation=pad_d),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )


class ScaleDeformConv(nn.Sequential):
    def __init__(self, in_channels, kernel_size, padding):
        super(ScaleDeformConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            DeformConv(in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )


class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=bias)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=bias)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


def MBConv(
        dim_in,
        dim_out,
        *,
        downsample,
        pd,
        expansion_rate=4,
        shrinkage_rate=0.25,
        dropout=0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


# f = ScaleDeformConv(in_channels=64, kernel_size=(7, 7), padding=3)
# x = torch.rand([4, 64, 56, 56])
# y = f(x)
# print(y.shape)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels=None, bilinear=True, linear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.linear = linear
        if linear:
            self.up_channel_1 = nn.Linear(in_channels, in_channels // 2, bias=False)
            self.up_channel_2 = nn.Linear(in_channels, in_channels // 2, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            self.conv2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        if self.linear:
            x1 = self.up_channel_1(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # b, h, w, c -> b, c, h, w
        else:
            x1 = self.conv1(x1)

        x = torch.cat([x2, x1], dim=1)

        # torch.Size([1, 512, 14, 14])
        # torch.Size([1, 256, 28, 28])
        # torch.Size([1, 128, 56, 56])

        if self.linear:
            x = self.up_channel_2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # b, h, w, c -> b, c, h, w
        else:
            x = self.conv2(x)

        return x
