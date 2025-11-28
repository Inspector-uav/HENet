import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Denhance(nn.Module):
    def __init__(self, out_channels):
        super(Denhance, self).__init__()
        self.cbr = ConvBNReLU(2, out_channels, ksize=3, pad=1, bias=False)
        self.conv1 = nn.Conv2d(2, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(2, out_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)

        cbr_output = self.cbr(scale)

        conv1_output = self.conv1(scale)
        # print('conv1_output.shape', conv1_output.shape)
        conv2_output = self.conv2(scale)
        conv2_output_t = conv2_output.transpose(2, 3)
        sigmoid_output = self.sigmoid(conv1_output * conv2_output_t)
        enhanced_feature = cbr_output * sigmoid_output

        return enhanced_feature


class Tenhance(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tenhance, self).__init__()
        # X方向的卷积
        self.conv3x1_d1 = nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0), dilation=1, bias=True)
        self.conv3x1_d2 = nn.Conv2d(in_channels, out_channels, (3, 1), padding=(2, 0), dilation=2, bias=True)
        self.conv3x1_d4 = nn.Conv2d(in_channels, out_channels, (3, 1), padding=(4, 0), dilation=4, bias=True)
        # Y方向的卷积
        self.conv1x3_d1 = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), dilation=1, bias=True)
        self.conv1x3_d2 = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 2), dilation=2, bias=True)
        self.conv1x3_d4 = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 4), dilation=4, bias=True)

    def forward(self, x):
        feature_size_x = x.shape[-1]
        feature_size_y = x.shape[-2]
        pooled_x = F.avg_pool2d(x, kernel_size=(1, feature_size_x))

        out_d1_x = self.conv3x1_d1(pooled_x)
        out_d2_x = self.conv3x1_d2(pooled_x)
        out_d4_x = self.conv3x1_d4(pooled_x)
        concatenated_x = torch.cat((out_d1_x, out_d2_x, out_d4_x), dim=-1)

        output_x = F.interpolate(concatenated_x, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        pooled_y = F.avg_pool2d(x, kernel_size=(feature_size_y, 1))

        out_d1_y = self.conv1x3_d1(pooled_y)
        out_d2_y = self.conv1x3_d2(pooled_y)
        out_d4_y = self.conv1x3_d4(pooled_y)
        concatenated_y = torch.cat((out_d1_y, out_d2_y, out_d4_y), dim=-2)

        output_y = F.interpolate(concatenated_y, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        output = output_x + output_y

        return output


class FIFM(nn.Module):
    def __init__(self, in_channels, out_channels):  # 256, 128
        super(FIFM, self).__init__()
        self.tenhance = Tenhance(in_channels, int(out_channels / 2))
        self.denhance = Denhance(int(out_channels / 2))

    def forward(self, t, d):
        enhanced_t = self.tenhance(t)
        enhanced_d = self.denhance(d)
        final_output = torch.cat([enhanced_t, enhanced_d], dim=1)
        return final_output
