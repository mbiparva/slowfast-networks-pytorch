from utils.config import cfg
import time
from abc import ABC, abstractmethod
import datetime
import torch

from data.data_container import DataContainer
from utils.miscellaneous import AverageMeter


class TVBase(ABC):
    def __init__(self, mode, meters, device):
        self.data_container = None
        assert mode in ('train', 'valid')
        self.mode, self.device = mode, device
        self.meters = {m: AverageMeter() for m in meters}

        self.create_dataset()

    def reset_meters(self):
        for m in self.meters.values():
            m.reset()

    def update_meters(self, **kwargs):
        for k, m in self.meters.items():
            try:
                m.update(kwargs[k])
            except KeyError:
                raise KeyError('Key {} is not defined in the dictionary'.format(k))

    def get_avg(self):
        for k, m in self.meters.items():
            yield k, m.avg

    def create_dataset(self):
        self.data_container = DataContainer(self.mode)

    def result_print(self, i, epoch, batch_time, started_time):
        print(
            '{4} {0} [{1}][{2}/{3}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Label Acc. {label.val:.4f} ({label.avg:.4f})'.format(
                self.mode.upper(), epoch, i, len(self.data_container.dataloader) - 1,
                str(datetime.timedelta(seconds=int(time.time() - started_time))), batch_time=batch_time,
                loss=self.meters['loss'],
                label=self.meters['label_accuracy']))

    @abstractmethod
    def set_net_mode(self, net):
        pass

    def generate_gt(self, annotation):
        return annotation['label'].to(self.device)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def evaluate(p, a):
        return (p.argmax(dim=1) == a).sum().item() / len(a)

    @abstractmethod
    def batch_main(self, net, x_slow, x_fast, annotation):
        pass

    def batch_loop(self, net, epoch, started_time):
        batch_time = AverageMeter()
        end = time.time()
        for i, (image, annotation) in enumerate(self.data_container.dataloader):

            x_slow = image[:, :, ::cfg.SLOWFAST.TAU, :, :].to(self.device)
            x_fast = image[:, :, ::cfg.SLOWFAST.TAU//cfg.SLOWFAST.ALPHA, :, :].to(self.device)

            results = self.batch_main(net, x_slow, x_fast, annotation)
            self.update_meters(**results)

            batch_time.update(time.time() - end)
            end = time.time()

            self.result_print(i, epoch, batch_time, started_time)
