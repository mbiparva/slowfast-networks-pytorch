import os
from utils.config import cfg

import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.optim as optim

from netwrapper.slow_pathway import resnet50_s
from netwrapper.fast_pathway import resnet50_f


# noinspection PyProtectedMember
class StepLRestart(optim.lr_scheduler._LRScheduler):
    """The same as StepLR, but this one has restart.
    """
    def __init__(self, optimizer, step_size, restart_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.restart_size = restart_size
        assert self.restart_size > self.step_size
        self.gamma = gamma
        super(StepLRestart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** ((self.last_epoch % self.restart_size) // self.step_size)
                for base_lr in self.base_lrs]


class TwoStreamNet(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.slow_net, self.fast_net = None, None
        self.criterion, self.optimizer, self.scheduler = None, None, None

        self.create_load(device)

        self.dropout = nn.Dropout(cfg.SLOWFAST.DP).to(device)
        self.fc = nn.Linear(4*512 + 4*512//cfg.SLOWFAST.ALPHA, cfg.NUM_CLASSES,
                            bias=False).to(device)
        # nn_init.normal_(self.fc.weight)
        # nn_init.xavier_normal_(self.fc.weight)
        nn_init.kaiming_normal_(self.fc.weight)

        self.setup_optimizer()

    def create_load(self, device):
        if cfg.PRETRAINED_MODE == 'Custom':
            self.create_net()
            self.load(cfg.PT_PATH)
        else:
            self.create_net()

        self.slow_net = self.slow_net.to(device)
        self.fast_net = self.fast_net.to(device)

    def create_net(self):
        self.slow_net = resnet50_s(**{
            'in_channels': cfg.CHANNEL_INPUT_SIZE,
            'num_classes': cfg.NUM_CLASSES,
            'alpha': cfg.SLOWFAST.ALPHA,
            'slow': 1,
            't2s_mul': cfg.SLOWFAST.T2S_MUL,
        })
        self.fast_net = resnet50_f(**{
            'in_channels': cfg.CHANNEL_INPUT_SIZE,
            'num_classes': cfg.NUM_CLASSES,
            'alpha': cfg.SLOWFAST.ALPHA,
            'slow': 0,
            't2s_mul': cfg.SLOWFAST.T2S_MUL,
        })

    def setup_optimizer(self):
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(params=self.parameters(),
                                   lr=cfg.TRAIN.LR,
                                   weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                   momentum=cfg.TRAIN.MOMENTUM,
                                   nesterov=cfg.TRAIN.NESTEROV)

        if cfg.TRAIN.SCHEDULER_MODE:
            if cfg.TRAIN.SCHEDULER_TYPE == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.TRAIN.SCHEDULER_STEP_MILESTONE,
                                                           gamma=0.1)
            elif cfg.TRAIN.SCHEDULER_TYPE == 'step_restart':
                self.scheduler = StepLRestart(self.optimizer, step_size=4, restart_size=8, gamma=0.1)
            elif cfg.TRAIN.SCHEDULER_TYPE == 'multi':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                milestones=cfg.TRAIN.SCHEDULER_MULTI_MILESTONE,
                                                                gamma=0.1)
            elif cfg.TRAIN.SCHEDULER_TYPE == 'lambda':
                def lr_lambda(e): return 1 if e < 5 else .5 if e < 10 else .1 if e < 15 else .01
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            elif cfg.TRAIN.SCHEDULER_TYPE == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5,
                                                                      cooldown=0,
                                                                      verbose=True)
            else:
                raise NotImplementedError

    def schedule_step(self, metric=None):
        if cfg.TRAIN.SCHEDULER_MODE:
            if cfg.TRAIN.SCHEDULER_TYPE in ['step', 'step_restart', 'multi', 'lambda']:
                self.scheduler.step()
            if cfg.TRAIN.SCHEDULER_TYPE == 'plateau':
                self.scheduler.step(metric['loss'].avg)

    def save(self, file_path, e):
        torch.save(self.state_dict(), os.path.join(file_path, '{:03d}.pth'.format(e)))

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def forward(self, x):
        x_slow, x_fast = x
        x_fast, laterals = self.fast_net(x_fast)
        x_slow = self.slow_net((x_slow, laterals))

        x = torch.cat([x_slow, x_fast], dim=1)

        x = self.dropout(x)
        x = self.fc(x)

        return x

    def loss_update(self, p, a, step=True):
        loss = self.criterion(p, a)

        if step:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()
