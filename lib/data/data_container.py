from utils.config import cfg

import os
import datetime

import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data
from data.data_set import UCF101
import data.spatial_transformation as Transformation

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def ds_worker_init_fn(worker_id):
    np.random.seed(datetime.datetime.now().microsecond + worker_id)


class DataContainer:
    def __init__(self, mode):
        self.dataset, self.dataloader = None, None
        self.mode = mode
        self.mode_cfg = cfg.get(self.mode.upper())

        self.create()

    def create(self):
        self.create_dataset()
        self.create_dataloader()

    def create_transform(self):
        h, w = cfg.SPATIAL_INPUT_SIZE
        assert h == w
        transformations_final = [
            Transformation.ToTensor(),
            Transformation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet-driven
        ]
        if self.mode == 'train':
            # transformations = [
            #     Transformation.CenterCrop(240),
            #     Transformation.Resize(h),
            # ]
            transformations = [
                # Transformation.RandomCornerCrop(240, crop_scale=(240, 224, 192, 168), border=0.25),
                Transformation.RandomCornerCrop(240, crop_scale=(0.66, 1.0), border=0.25),
                Transformation.Resize(h),   # This is necessary for async-resolution streaming
                Transformation.RandomHorizontalFlip()
            ]
        elif self.mode == 'valid':
            transformations = [
                Transformation.CenterCrop(240),
                Transformation.Resize(h),
            ]
        else:
            raise NotImplementedError

        return Transformation.Compose(
            [Transformation.ToPILImage('RGB')] + transformations + transformations_final  # since cv2 loads into np
        )

    def create_dataset(self):
        spatial_transform = self.create_transform()

        if cfg.DATASET_NAME == 'UCF101':
            self.dataset = UCF101(self.mode, spatial_transform)

    def create_dataloader(self):
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.mode_cfg.BATCH_SIZE,
                                     shuffle=self.mode_cfg.SHUFFLE,
                                     num_workers=4,
                                     pin_memory=True,
                                     drop_last=True,
                                     worker_init_fn=ds_worker_init_fn,
                                     )
