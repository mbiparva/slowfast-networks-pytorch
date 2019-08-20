import _init_lib_path

import os
from datetime import datetime
import datetime as dt
import time

from utils.config import cfg
from epoch_loop import EpochLoop

import argparse
from utils.config_file_handling import cfg_from_file, cfg_from_list
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4)
cfg.TRAINING = False
cfg.VALIDATING = True
cfg.PRETRAINED_MODE = 'Custom'


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Testing the network')

    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir',
                        help='dataset directory', type=str, required=False)
    parser.add_argument('-e', '--experiment-dir', dest='experiment_dir',
                        help='a directory used to write experiment results', type=str, required=False)
    parser.add_argument('-i', '--pre-trained-id', dest='pt_id',
                        help='the pre-trained network id that you want to load for testing', type=str, required=True)
    parser.add_argument('-p', '--pre-trained-epoch', dest='pt_epoch',
                        help='the epoch at which a snapshot for the id is taken', type=str, required=True)
    parser.add_argument('-u', '--use-gpu', dest='use_gpu',
                        help='whether to use gpu for the net inference', type=int, required=False)
    parser.add_argument('-g', '--gpu-id', dest='gpu_id',
                        help='gpu id to use', type=int, required=False)
    parser.add_argument('-c', '--cfg', dest='cfg_file',
                        help='optional config file to override the defaults', default=None, type=str)
    parser.add_argument('-s', '--set', dest='set_cfg',
                        help='set config arg parameters', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def set_positional_cfg(args_in):
    args_list = []
    for n, a in args_in.__dict__.items():
        if a is not None and n not in ['cfg_file', 'set_cfg']:
            args_list += [n, a]
    return args_list


def main():
    epoch_loop = EpochLoop()

    try:
        epoch_loop.main()
    except KeyboardInterrupt:
        print('*** The experiment is terminated by a keyboard interruption')


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfg is not None:
        cfg_from_list(args.set_cfg)

    cfg_from_list(set_positional_cfg(args))     # input arguments override cfg files and defaults

    cfg.PT_PATH = os.path.join(cfg.EXPERIMENT_DIR,
                               'snapshot_{}_{}'.format(cfg.DATASET_NAME, cfg.NET_ARCH),
                               args.pt_id,
                               '{:03}.pt'.format(args.pt_epoch))

    print('configuration file cfg is loaded for testing ...')
    pp.pprint(cfg)

    started_time = time.time()
    print('*** started @', datetime.now())
    main()
    length = time.time() - started_time
    print('*** ended @', datetime.now())
    print('took', dt.timedelta(seconds=int(length)))
