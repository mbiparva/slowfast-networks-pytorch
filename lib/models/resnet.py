import torch
import torch.nn as nn
import torch.nn.init as nn_init
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
from torch.nn import Parameter

__all__ = ['ResNet3D', 'resnet50']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


# =============================
# ********* 3D ResNet *********
# =============================
def conv1x3x3(in_planes, out_planes, stride=1):
    """1x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = conv1x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def init_temporal(self, strategy):
        raise NotImplementedError


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bias=False, head_conv=1):
        super().__init__()

        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")

        self.conv2 = nn.Conv3d(planes, planes,
                               kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=bias)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, **kwargs):
        super().__init__()
        in_channels, num_classes = kwargs['in_channels'], kwargs['num_classes']
        self.alpha = kwargs['alpha']
        self.slow = kwargs['slow']  # slow->1 else fast->0
        self.t2s_mul = kwargs['t2s_mul']
        self.inplanes = (64 + 64//self.alpha*self.t2s_mul) if self.slow else 64//self.alpha
        self.conv1 = nn.Conv3d(in_channels, 64//(1 if self.slow else self.alpha),
                               kernel_size=(1 if self.slow else 5, 7, 7),
                               stride=(1, 2, 2), padding=(0 if self.slow else 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64//(1 if self.slow else self.alpha))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64//(1 if self.slow else self.alpha), layers[0],
                                       head_conv=1 if self.slow else 3)
        self.layer2 = self._make_layer(block, 128//(1 if self.slow else self.alpha), layers[1], stride=2,
                                       head_conv=1 if self.slow else 3)
        self.layer3 = self._make_layer(block, 256//(1 if self.slow else self.alpha), layers[2], stride=2,
                                       head_conv=3)
        self.layer4 = self._make_layer(block, 512//(1 if self.slow else self.alpha), layers[3], stride=2,
                                       head_conv=3)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn_init.normal_(m.weight)
                # nn_init.xavier_normal_(m.weight)
                nn_init.kaiming_normal_(m.weight)
                if m.bias:
                    nn_init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
                nn_init.constant_(m.weight, 1)
                nn_init.constant_(m.bias, 0)

    def forward(self, x):
        raise NotImplementedError('use each pathway network\' forward function')

    def _make_layer(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, head_conv=head_conv))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, head_conv=head_conv))

        self.inplanes += self.slow * block.expansion * planes // self.alpha * self.t2s_mul

        return nn.Sequential(*layers)


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet3D(Bottleneck3D, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pt = model_zoo.load_url(model_urls['resnet50'])
        pt.pop('conv1.weight')
        pt.pop('fc.weight')
        pt.pop('fc.bias')
        model_dict = model.state_dict()
        model_dict.update(pt)
        model.load_state_dict(model_dict)

    return model
