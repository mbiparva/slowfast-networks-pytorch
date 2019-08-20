from utils.config import cfg

import models.resnet
import torch.nn as nn
import torch.nn.functional as F


class FastNet(models.resnet.ResNet3D):
    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)
        self.l_maxpool = nn.Conv3d(64//self.alpha, 64//self.alpha*self.t2s_mul,
                                   kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False, padding=(2, 0, 0))
        self.l_layer1 = nn.Conv3d(4*64//self.alpha, 4*64//self.alpha*self.t2s_mul,
                                  kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False, padding=(2, 0, 0))
        self.l_layer2 = nn.Conv3d(8*64//self.alpha, 8*64//self.alpha*self.t2s_mul,
                                  kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False, padding=(2, 0, 0))
        self.l_layer3 = nn.Conv3d(16*64//self.alpha, 16*64//self.alpha*self.t2s_mul,
                                  kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False, padding=(2, 0, 0))
        self.init_params()

    def forward(self, x):
        laterals = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        laterals.append(self.l_maxpool(x))

        x = self.layer1(x)
        laterals.append(self.l_layer1(x))

        x = self.layer2(x)
        laterals.append(self.l_layer2(x))

        x = self.layer3(x)
        laterals.append(self.l_layer3(x))

        x = self.layer4(x)

        x = F.adaptive_avg_pool3d(x, 1)
        x = x.view(-1, x.size(1))

        return x, laterals


def resnet50_f(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = FastNet(models.resnet.Bottleneck3D, [3, 4, 6, 3], **kwargs)

    return model
