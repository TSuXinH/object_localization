import torch
import torch.nn as nn
from collections import OrderedDict


# residual block
class residual_block(nn.Module):
    def __init__(self, channels):
        super(residual_block, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=1),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        residual = x
        x = self.layer1(x)
        x = self.layer2(x)
        out = x + residual
        return out


# YOLO v1 body
class YOLO_v1(nn.Module):
    # the original image shape: 224 * 224
    def __init__(self):
        super(YOLO_v1, self).__init__()
        self.basic_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)  # shape: 32 * 112 * 112
        )
        self.res_layer1 = self.make_layer([32, 64], 1)  # shape: 64 * 56 * 56
        self.res_layer2 = self.make_layer([64, 128], 8)  # shape: 128 * 28 * 28
        self.res_layer3 = self.make_layer([128, 256], 8)  # shape: 256 * 14 * 14
        self.res_layer4 = self.make_layer([256, 512], 4)  # shape: 512 * 7 * 7
        self.end_layer1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.Sigmoid()
        )
        self.end_layer2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Sigmoid()
        )
        self.end_layer3 = nn.Conv2d(32, 5, kernel_size=1)

    def forward(self, x):
        x = self.basic_layer(x)
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        x = self.res_layer4(x)
        x = self.end_layer1(x)
        x = self.end_layer2(x)
        x = self.end_layer3(x)
        out = x.contiguous().view(-1, 7, 7, 5)
        return out

    def make_layer(self, channels, blocks):
        layer = [
            # maintain the shape of feature map
            ('convolution', nn.Conv2d(channels[0], channels[1],
                                      kernel_size=3, stride=2, padding=1, bias=False)),
            ('batch normalization', nn.BatchNorm2d(channels[1])),
            ('ReLU', nn.LeakyReLU(0.1))
        ]
        r_channels = channels[:: -1]
        for num in range(blocks):
            layer.append(('residual_{}'.format(num + 1), residual_block(r_channels)))
        return nn.Sequential(OrderedDict(layer))
