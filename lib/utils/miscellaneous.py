# This is meant to contain miscellaneous functions and routines
# such as showing, saving, loading images and results.
import sys
import os
from PIL import Image
import numpy as np


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val, self.avg, self.sum, self.count = (0,)*4
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_size(path, image_size=(342, 256)):
    check_all = False
    outliers = []
    for m, n in enumerate(os.listdir(path)):
        dir_name = os.path.join(path, n)
        if os.path.isfile(dir_name):
            continue
        if check_all:
            for k, i in enumerate(os.listdir(dir_name)):
                image = Image.open(os.path.join(dir_name, i))
                assert image_size == image.size, '{2}:{0}|{1}'.format(image_size, image.size, n)
                print('{2}/{3}:{0}/{1}'.format(k, len(os.listdir(dir_name)), m, len(os.listdir(path))))
        else:
            dir_content = os.listdir(dir_name)
            image_name = np.random.choice(dir_content)
            image_path = os.path.join(dir_name, image_name)
            image = Image.open(image_path)
            if not image_size == image.size:
                print('---one outlier detected---')
                outliers.append('{2}:{0}|{1}'.format(image_size, image.size, n))
            print('{0}/{1}:{2}'.format(m, len(os.listdir(path)), n))
    for o in outliers:
        print(o)


if __name__ == '__main__':
    if sys.argv[1] == 'print_size':
        print_size(sys.argv[2], (int(sys.argv[3]), int(sys.argv[4])))
