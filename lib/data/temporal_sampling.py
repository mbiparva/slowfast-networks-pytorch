from utils.config import cfg

import math
import numpy as np
import os


class TemporalSampler:
    def __init__(self, frame_sampling_method):
        self.frame_sampling_method = frame_sampling_method
        self.nf = cfg.NFRAMES_PER_VIDEO if self.frame_sampling_method != 'f25' else 25

    def frame_sampler(self, in_nframes):
        if self.frame_sampling_method == 'uniform':
            num_frames = max(1, self.nf-1)
            sample_rate = max(in_nframes // num_frames, 1)
            frame_samples = np.arange(0, in_nframes, sample_rate)

        elif self.frame_sampling_method == 'temporal_stride':
            frame_samples = np.arange(0, in_nframes, cfg.TEMPORAL_STRIDE[0])
            if len(frame_samples) < self.nf:
                frame_samples = np.linspace(0, in_nframes, self.nf, endpoint=False).astype(np.int)

        elif self.frame_sampling_method == 'random':
            frame_samples = np.random.permutation(in_nframes)

        elif self.frame_sampling_method == 'temporal_stride_random':
            temporal_stride = np.random.randint(cfg.TEMPORAL_STRIDE[0], cfg.TEMPORAL_STRIDE[1])
            frame_samples = np.arange(0, in_nframes, temporal_stride)

        elif self.frame_sampling_method == 'f25':
            frame_samples = np.linspace(0, in_nframes, 25, endpoint=False).round().tolist()

        else:
            raise NotImplementedError

        # check the under or over frame sample list length.
        if len(frame_samples) < self.nf:
            add_frames, difference = 0, self.nf - len(frame_samples)
            while difference > 0:
                next_len = len(frame_samples)
                add_samples = np.linspace(cfg.TEMPORAL_INPUT_SIZE / 4, in_nframes - cfg.TEMPORAL_INPUT_SIZE / 4,
                                          next_len, endpoint=False)
                add_samples = add_samples.round().tolist()
                add_samples = add_samples[np.random.randint(1, len(add_samples), 1)[0]]
                if add_frames > 20:
                    frame_samples = np.linspace(cfg.TEMPORAL_INPUT_SIZE / 4, in_nframes - cfg.TEMPORAL_INPUT_SIZE / 4,
                                                self.nf, endpoint=False)
                    frame_samples = frame_samples.round().tolist()
                else:
                    frame_samples = np.append(frame_samples, add_samples)
                difference = self.nf - len(frame_samples)
                add_frames += 1
            frame_samples = np.sort(frame_samples)
        elif len(frame_samples) > self.nf:
            start = np.random.randint(0, len(frame_samples)-self.nf)
            frame_samples = frame_samples[start:start+self.nf]
        else:
            pass

        if self.frame_sampling_method == 'random':
            frame_samples = np.sort(frame_samples)

        assert (frame_samples == np.sort(frame_samples)).all()  # ensure the frame numbers are sorted

        return frame_samples
