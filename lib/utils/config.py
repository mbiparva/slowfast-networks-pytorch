"""Config file setting hyperparameters

This file specifies default config options. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.
"""

from easydict import EasyDict as edict
import os
import datetime
import socket

__C = edict()
cfg = __C   # from config.py import cfg


# ================
# GENERAL
# ================

# Set modes
__C.TRAINING = True
__C.VALIDATING = True

# Root directory of project
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data directory
# TODO: take care of this before releasing the code
if socket.gethostname() == 'mahd-ubuntu':
    __C.DATASET_DIR = '/media/mahd/SamSSD/myImp_survived/AttentionFusionMultiSpeedNets/datasets'
else:
    __C.DATASET_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, 'dataset'))

# Model directory
__C.MODELS_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, 'lib', 'models'))

# Experiment directory
__C.EXPERIMENT_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, 'experiment'))

# Set meters to use for experimental evaluation
__C.METERS = ['loss', 'label_accuracy']

# Use GPU
__C.USE_GPU = True

# Default GPU device id
__C.GPU_ID = 0

# Number of epochs
__C.NUM_EPOCH = 40

# Dataset name
__C.DATASET_NAME = ('UCF101', )[0]

if __C.DATASET_NAME == 'UCF101':
    __C.SPLIT_NO = 1

    # Number of categories
    __C.NUM_CLASSES = 101

    __C.DATASET_ROOT = os.path.join(__C.DATASET_DIR, __C.DATASET_NAME)

# Normalize database samples according to some mean and std values
__C.DATASET_NORM = True

# Input data size
__C.SPATIAL_INPUT_SIZE = (112, 112)
__C.CHANNEL_INPUT_SIZE = 3

# Set parameters for snapshot and verbose routines
__C.MODEL_ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
__C.SNAPSHOT = True
__C.SNAPSHOT_INTERVAL = 5
__C.VERBOSE = True
__C.VERBOSE_INTERVAL = 10
__C.VALID_INTERVAL = 1

# Network Architecture
__C.NET_ARCH = ('resnet', )[0]

# Pre-trained network
__C.PRETRAINED_MODE = (None, 'Custom')[0]

# Path to the pre-segmentation network
__C.PT_PATH = os.path.join(__C.EXPERIMENT_DIR, 'snapshot', '20181010_124618_219443', '079.pt')

# =============================
# Spatiotemporal ResNet options
# =============================
__C.RST = edict()

__C.FRAME_SAMPLING_METHOD = ('uniform', 'temporal_stride', 'random', 'temporal_stride_random')[1]
__C.NFRAMES_PER_VIDEO = 64  # T x tau
__C.TEMPORAL_STRIDE = (1, 25)
__C.FRAME_RANDOMIZATION = False

# =============================
# SlowFast ResNet options
# =============================
__C.SLOWFAST = edict()

# T = NFRAMES_PER_VIDEO // TAU
__C.SLOWFAST.TAU = 16
__C.SLOWFAST.ALPHA = 8
__C.SLOWFAST.T2S_MUL = 2
__C.SLOWFAST.DP = 0.5

# ================
# Training options
# ================
if __C.TRAINING:
    __C.TRAIN = edict()

    # Images to use per minibatch
    __C.TRAIN.BATCH_SIZE = 32

    # Shuffle the dataset
    __C.TRAIN.SHUFFLE = True

    # Learning parameters are set below
    __C.TRAIN.LR = 1e-3
    __C.TRAIN.WEIGHT_DECAY = 1e-5
    __C.TRAIN.MOMENTUM = 0.90
    __C.TRAIN.NESTEROV = False
    __C.TRAIN.SCHEDULER_MODE = False
    __C.TRAIN.SCHEDULER_TYPE = ('step', 'step_restart', 'multi', 'lambda', 'plateau')[0]
    __C.TRAIN.SCHEDULER_STEP_MILESTONE = 10
    __C.TRAIN.SCHEDULER_MULTI_MILESTONE = [10]

# ================
# Validation options
# ================
if __C.VALIDATING:
    __C.VALID = edict()

    # Images to use per minibatch
    __C.VALID.BATCH_SIZE = __C.TRAIN.BATCH_SIZE

    # Shuffle the dataset
    __C.VALID.SHUFFLE = False
