import torch
from torch import nn
import numpy as np
import torch.nn.functional as F



class full_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_norm=True):
        super(full_conv, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('conv', nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding))
        if use_norm == True:
            self.net.add_module('norm', nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))
        self.net.add_module('relu', nn.LeakyReLU(0.2))

    def forward(self, input):
        result = self.net(input)
        return result

class conv_relu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, layer_num=1):
        super(conv_relu, self).__init__()
        self.net = nn.Sequential()

        self.net.add_module('full_conv1', full_conv(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding))
        for i in range(1, layer_num, 1):
            self.net.add_module(f'full_conv{i+1}', full_conv(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding))

    def forward(self, input):
        result = self.net(input)
        return result


class feature_encoding(nn.Module):
    def __init__(self):
        super(feature_encoding, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('full_conv1', full_conv(3, 32, use_norm=False))
        self.net.add_module('full_conv2', full_conv(32, 32, use_norm=False))
        self.net.add_module('full_conv3', full_conv(32, 32, use_norm=False))
        self.net.add_module('full_conv4', full_conv(32, 32, use_norm=False))
        self.net.add_module('squeeze_excitation', squeeze_excitation_layer(32, 32, 2))
        self.net.add_module('full_conv5', full_conv(32, 3, use_norm=False))

    def forward(self, input):
        result = self.net(input)
        return result


class squeeze_excitation_layer(nn.Module):
    def __init__(self, in_ch, out_ch, middle):
        super(squeeze_excitation_layer, self).__init__()
        self.out_dim = out_ch
        self.net = nn.Sequential()
        self.net.add_module('Affine1', nn.Linear(in_ch, middle))
        self.net.add_module('relu', nn.ReLU())
        self.net.add_module('Affine2', nn.Linear(middle, out_ch))
        self.net.add_module('sigmoid', nn.Sigmoid())

    def forward(self, input):
        squeeze = nn.functional.adaptive_avg_pool2d(input, (1, 1))
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.net(squeeze).reshape(-1, self.out_dim, 1, 1)
        scale = input * excitation
        return scale

# 长宽变两倍
class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(deconv2d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class feature_fusion(nn.Module):
    def __init__(self):
        super(feature_fusion, self).__init__()
        self.layer_conv1 = full_conv(21, 7, kernel_size=3, padding=1)
        self.layer_conv2 = full_conv(21, 7, kernel_size=5, padding=2)
        self.layer_conv3 = full_conv(21, 7, kernel_size=7, padding=3)
        self.layer_selective = selective_kernel(4, 7)
        self.layer_out = nn.Conv2d(7, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_map, resize1, resize2, resize3, resize4, resize5):
        concat = torch.cat([feature_map, resize1, resize2, resize3, resize4, resize5], dim=1)
        sk_conv1 = self.layer_conv1(concat)
        sk_conv2 = self.layer_conv2(concat)
        sk_conv3 = self.layer_conv3(concat)
        sk_out = self.layer_selective(sk_conv1, sk_conv2, sk_conv3)
        return self.layer_out(sk_out)





class selective_kernel(nn.Module):
    def __init__(self, middle, out_ch):
        super(selective_kernel, self).__init__()
        self.out_ch = out_ch
        self.middle = middle

        self.affine1 = nn.Linear(out_ch, middle)
        self.affine2 = nn.Linear(middle, out_ch)


    def forward(self, sk_conv1, sk_conv2, sk_conv3):
        sum_u = sk_conv1 + sk_conv2 + sk_conv3
        squeeze = nn.functional.adaptive_avg_pool2d(sum_u, (1, 1))
        squeeze = squeeze.view(squeeze.size(0), -1)
        z = self.affine1(squeeze)
        z = F.relu(z)
        a1 = self.affine2(z).reshape(-1, self.out_ch, 1, 1)
        a2 = self.affine2(z).reshape(-1, self.out_ch, 1, 1)
        a3 = self.affine2(z).reshape(-1, self.out_ch, 1, 1)

        before_softmax = torch.cat([a1, a2, a3], dim=1)
        after_softmax = F.softmax(before_softmax, dim=1)
        a1 = after_softmax[:, 0:self.out_ch, :, :]
        a1.reshape(-1, self.out_ch, 1, 1)

        a2 = after_softmax[:, self.out_ch:2*self.out_ch, :, :]
        a2.reshape(-1, self.out_ch, 1, 1)
        a3 = after_softmax[:, 2*self.out_ch:3*self.out_ch, :, :]
        a3.reshape(-1, self.out_ch, 1, 1)

        select_1 = sk_conv1 * a1
        select_2 = sk_conv2 * a2
        select_3 = sk_conv3 * a3

        return select_1 + select_2 + select_3


def initialize_weights(net):
    for m in net.modules():
        try:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        except Exception as e:
            pass


