from utils.config import cfg

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from data.temporal_sampling import TemporalSampler


class UCF101(Dataset):
    def __init__(self, mode, spatial_trans):
        self.mode = mode
        self.dataset_path = os.path.join(cfg.DATASET_ROOT, 'training' if self.mode == 'train' else 'validation')
        self.spatial_trans = spatial_trans

        self.temporal_sampler = TemporalSampler(cfg.FRAME_SAMPLING_METHOD)

        self.file_names, file_labels, = [], []
        for l in sorted(os.listdir(self.dataset_path)):
            for f in os.listdir(os.path.join(self.dataset_path, l)):
                self.file_names.append(os.path.join(self.dataset_path, l, f))
                file_labels.append(l)

        self.l2i = {l: i for i, l in enumerate(sorted(set(file_labels)))}
        self.file_labels = np.asarray(list(map(lambda lab: self.l2i[lab], file_labels)), dtype=np.int)

    def __getitem__(self, index):
        f_name, f_label = self.file_names[index], self.file_labels[index]

        f_frames, f_height, f_width = self.video_info_retrieval(f_name)

        f_frame_list = self.temporal_sampler.frame_sampler(f_frames)

        frame_bank = self.load_frames(f_name, f_frame_list)

        frames_transformed = []

        self.spatial_trans.randomize_parameters()
        for i in frame_bank:
            if cfg.FRAME_RANDOMIZATION:
                self.spatial_trans.randomize_parameters()
            frames_transformed.append(self.spatial_trans(i))

        frames_packed = self.pack_frames(frames_transformed)

        return frames_packed, {'file_path': f_name, 'file_name': f_name.split('/')[-1], 'nframes': f_frames, 'label': f_label}

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def count_frames_accurately(vid_cap):
        frames = 0
        retaining, frame_data = vid_cap.read()

        while retaining:
            frames += 1
            retaining, frame_data = vid_cap.read()

        assert frames > 0, 'video file is corrupted, could not count frames'
        return frames

    def video_info_retrieval(self, file_name):
        vid_cap = cv2.VideoCapture(file_name)
        f_c = self.count_frames_accurately(vid_cap)  # WE MUST USE THE MANUAL VERSION, IT IS ERROR PRONE.
        f_h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_cap.release()

        return f_c, f_h, f_w

    @staticmethod
    def load_frames(fname, frame_list):
        vid_cap = cv2.VideoCapture(fname)

        retaining, frame_data = vid_cap.read()
        assert retaining, 'the video clip is initially empty, very odd, maybe it is corrupted'
        frame_count = 0
        frame_bank = []
        frame_list_iter = iter(frame_list)
        f = next(frame_list_iter)
        break_out = False

        while retaining:
            frame_data = frame_data[:, :, ::-1]  # OpenCV loads in BGR
            while f == frame_count:
                frame_bank.append(frame_data)
                try:
                    f = next(frame_list_iter)
                except StopIteration:
                    break_out = True
                    break

            if break_out:
                break

            retaining, frame_data = vid_cap.read()
            frame_count += 1

        vid_cap.release()

        return frame_bank

    @staticmethod
    def pack_frames(frames):
        frames_out = torch.stack(frames).transpose(1, 0)

        return frames_out
