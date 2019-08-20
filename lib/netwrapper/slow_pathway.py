from utils.config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.resnet


class SlowNet(models.resnet.ResNet3D):
    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)
        self.init_params()

    def forward(self, x):
        x, laterals = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.cat([x, laterals[0]], dim=1)
        x = self.layer1(x)

        x = torch.cat([x, laterals[1]], dim=1)
        x = self.layer2(x)

        x = torch.cat([x, laterals[2]], dim=1)
        x = self.layer3(x)

        x = torch.cat([x, laterals[3]], dim=1)
        x = self.layer4(x)

        x = F.adaptive_avg_pool3d(x, 1)
        x = x.view(-1, x.size(1))

        return x


def resnet50_s(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = SlowNet(models.resnet.Bottleneck3D, [3, 4, 6, 3], **kwargs)

    return model
