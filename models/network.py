import torch
from torch import nn
import numpy as np
from .blocks import deconv2d_bn, conv_relu
import torch.nn.functional as F
from .blocks import feature_encoding, feature_fusion, initialize_weights



class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.layer1_conv = conv_relu(6, 32, layer_num=4)
        self.layer2_conv = conv_relu(32, 64, layer_num=4)
        self.layer3_conv = conv_relu(64, 128, layer_num=4)
        self.layer4_conv = conv_relu(128, 256, layer_num=4)
        self.layer5_conv = conv_relu(256, 512, layer_num=4)

        self.layer6_conv = conv_relu(512, 256, layer_num=3)
        self.layer7_conv = conv_relu(256, 128, layer_num=3)
        self.layer8_conv = conv_relu(128, 64, layer_num=3)
        self.layer9_conv = conv_relu(64, 32, layer_num=3)

        self.layer10_conv = nn.Conv2d(32, 3, kernel_size=1, stride=1)

        self.deconv1 = deconv2d_bn(512, 256)
        self.deconv2 = deconv2d_bn(256, 128)
        self.deconv3 = deconv2d_bn(128, 64)
        self.deconv4 = deconv2d_bn(64, 32)


    def forward(self, input):
        conv1 = self.layer1_conv(input)
        pool1 = F.max_pool2d(conv1, 2)
        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)
        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)
        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)
        conv5 = self.layer5_conv(pool4)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1, conv4], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3, conv2], dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4, conv1], dim=1)
        conv9 = self.layer9_conv(concat4)

        return self.layer10_conv(conv9)



class PRIDNet(nn.Module):
    def __init__(self):
        super(PRIDNet, self).__init__()
        self.feature_encoding = feature_encoding()
        self.unet = Unet()
        self.name = 'PRIDNet'

        self.pool1 = nn.AvgPool2d(kernel_size=1, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pool4 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.pool5 = nn.AvgPool2d(kernel_size=16, stride=16)

        self.feature_fusion = feature_fusion()
        self.layer_out = nn.Tanh()

        initialize_weights(self)


    def forward(self, input):
        feature_map = self.feature_encoding(input)
        feature_map2 = torch.cat([input, feature_map], dim=1)


        pool1 = self.pool1(feature_map2)
        pool2 = self.pool2(feature_map2)
        pool3 = self.pool3(feature_map2)
        pool4 = self.pool4(feature_map2)
        pool5 = self.pool5(feature_map2)

        unet1 = self.unet(pool1)
        unet2 = self.unet(pool2)
        unet3 = self.unet(pool3)
        unet4 = self.unet(pool4)
        unet5 = self.unet(pool5)


        resize1 = F.interpolate(unet1, size=(unet1.shape[2], unet1.shape[3]), mode='bilinear', align_corners=True)
        resize2 = F.interpolate(unet2, size=(unet1.shape[2], unet1.shape[3]), mode='bilinear', align_corners=True)
        resize3 = F.interpolate(unet3, size=(unet1.shape[2], unet1.shape[3]), mode='bilinear', align_corners=True)
        resize4 = F.interpolate(unet4, size=(unet1.shape[2], unet1.shape[3]), mode='bilinear', align_corners=True)
        resize5 = F.interpolate(unet5, size=(unet1.shape[2], unet1.shape[3]), mode='bilinear', align_corners=True)


        out = self.feature_fusion(feature_map2, resize1, resize2, resize3, resize4, resize5)
        return self.layer_out(out)











if __name__ == '__main__':
    import cv2

    img = cv2.imread('D://CV//Denoise//dataset//train//noise//00021.jpg').transpose(2, 0, 1) / 255
    img = img[None, :]
    img = torch.tensor((img), dtype=torch.float)

    model = PRIDNet()
    Q = model(img)
    Q = (Q.detach().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    print(Q)

    cv2.imshow('dq', Q)
    cv2.waitKey(0)

