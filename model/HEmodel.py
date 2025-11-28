import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import BatchNorm2d
from torch.nn.utils.spectral_norm import SpectralNorm


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s


class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
                 bias=True, use_relu=True, leaky_relu=False, use_bn=True, frozen=False, spectral_norm=False,
                 prelu=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=bias)
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            elif spectral_norm:
                self.bn = SpectralNorm(nOut)
            else:
                self.bn = BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            if leaky_relu is True:
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif prelu is True:
                self.act = nn.PReLU(nOut)
            else:
                self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channel: int, ratio: int = 4) -> None:
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(True),
            nn.Linear(channel // ratio, channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y = self.max_pool(x).view(b, c)

        y = self.fc(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        out = torch.mul(x, y)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, channel, kernel_size: int = 7) -> None:
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        max_out1, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv1(max_out1)
        weight_map = self.sigmoid(x)

        return weight_map


class HEM(nn.Module):
    def __init__(self, in_, out_, ratio: int = 4):
        super(HEM, self).__init__()
        self.conv1 = ConvBNReLU(in_, out_, 3, pad=1)
        self.conv2 = ConvBNReLU(in_, out_, 3, pad=1)
        self.conv3 = ConvBNReLU(in_, out_, 3, pad=1)

        self.sa1 = SpatialAttention(out_, kernel_size=3)
        self.sa2 = SpatialAttention(out_, kernel_size=3)
        self.ca1 = ChannelAttention(out_, ratio)
        self.ca2 = ChannelAttention(out_, ratio)

        self.conv_reduce = nn.Conv2d(2 * out_, out_, kernel_size=1)

        self.conv_3_1 = nn.Sequential(nn.Conv2d(out_, out_, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(out_),
                                      nn.ReLU())
        self.conv_1 = nn.Conv2d(out_, out_, kernel_size=1)
        self.conv_3_2 = nn.Sequential(nn.Conv2d(out_, out_, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(out_),
                                      nn.ReLU())
        self.relu = nn.ReLU()
        self.conv_x = nn.Sequential(nn.Conv2d(out_, out_, kernel_size=3, dilation=1, padding=1),
                                    nn.BatchNorm2d(out_),
                                    nn.Dropout(0.2))

    def forward(self, r, t_d):
        rgb1 = self.conv1(r)
        t_d1 = self.conv3(t_d)
        e1 = r + t_d
        rgb2 = self.conv2(e1) * rgb1
        t_d2 = self.conv2(e1) * t_d1

        map_t_depth = self.sa1(t_d2)
        rgb_output = self.ca1(rgb1.mul(map_t_depth))

        map_rgb = self.sa2(rgb2)
        t_depth_put = self.ca2(t_d1.mul(map_rgb))

        out = torch.cat((rgb_output, t_depth_put), dim=1)

        out = self.conv_reduce(out)

        enhance = self.conv_3_1(out)
        enhance = self.conv_1(enhance)
        enhance = self.conv_3_2(enhance)

        out = self.conv_x(out + enhance)

        return out
