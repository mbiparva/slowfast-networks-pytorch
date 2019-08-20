from utils.config import cfg
from utils.config_file_handling import cfg_to_file

import os
import time
from trainer import Trainer
from validator import Validator

import torch
from tensorboardX import SummaryWriter
from netwrapper.two_stream_net import TwoStreamNet

started_time = time.time()


class EpochLoop:
    def __init__(self):
        self.trainer, self.validator = None, None
        self.device, self.net = None, None
        self.logger_writer = None

        self.setup_gpu()

    def setup_gpu(self):
        cuda_device_id = cfg.GPU_ID
        if cfg.USE_GPU and torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(cuda_device_id))
        else:
            self.device = torch.device('cpu')

    def setup_logger(self):
        logger_dir = os.path.join(cfg.EXPERIMENT_DIR,
                                  'logger_{}_{}'.format(cfg.DATASET_NAME, cfg.NET_ARCH),
                                  cfg.MODEL_ID)
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        self.logger_writer = SummaryWriter(logger_dir)
        cfg_to_file(cfg, logger_dir, '{}_cfg'.format(cfg.MODEL_ID))

    def logger_update(self, e, mode):
        if e == 0:
            self.setup_logger()
        if mode == 'train' and self.trainer:
            for k, m_avg in self.trainer.get_avg():
                self.logger_writer.add_scalar('{}/{}'.format('train', k), m_avg, e)
        if mode == 'valid' and self.validator:
            for k, m_avg in self.validator.get_avg():
                self.logger_writer.add_scalar('{}/{}'.format('valid', k), m_avg, e)

    def check_if_save_snapshot(self, e):
        if cfg.SNAPSHOT and (e + 1) % cfg.SNAPSHOT_INTERVAL == 0:
            file_path = os.path.join(cfg.EXPERIMENT_DIR,
                                     'snapshot_{}_{}'.format(cfg.DATASET_NAME, cfg.NET_ARCH),
                                     cfg.MODEL_ID)
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            self.net.save(file_path, e)

    def check_if_validating(self, e):
        if cfg.VALIDATING and (e + 1) % cfg.VALID_INTERVAL == 0:
            self.validator_epoch_loop(e)

    def main(self):
        self.create_sets()
        self.setup_net()
        self.run()

    def create_sets(self):
        self.trainer = Trainer('train', cfg.METERS, self.device) if cfg.TRAINING else None
        self.validator = Validator('valid', cfg.METERS, self.device) if cfg.VALIDATING else None

    def setup_net(self):
        self.net = TwoStreamNet(self.device)

    def run(self):
        if cfg.TRAINING:
            self.trainer_epoch_loop()
        elif cfg.VALIDATING:
            self.validator_epoch_loop(0)
        elif cfg.TESTING:
            raise NotImplementedError('TESTING mode is not implemented yet')
        else:
            raise NotImplementedError('One of {TRAINING, VALIDATING, TESTING} must be set to True')

    def trainer_epoch_loop(self):
        for e in range(cfg.NUM_EPOCH):
            self.trainer.set_net_mode(self.net)

            self.trainer.reset_meters()

            self.trainer.batch_loop(self.net, e, started_time)

            self.check_if_save_snapshot(e)

            self.check_if_validating(e)

            self.logger_update(e, mode='train')

            self.net.schedule_step(metric=self.validator.meters)

    def validator_epoch_loop(self, e):
        self.validator.set_net_mode(self.net)

        self.validator.reset_meters()

        self.validator.batch_loop(self.net, e, started_time)

        self.logger_update(e, mode='valid')
