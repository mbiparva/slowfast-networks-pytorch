import collections
import numbers
import random

import torch
import torchvision.transforms
from torchvision.transforms import functional
import numpy as np
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class Compose(torchvision.transforms.Compose):
    """Compose class that has randomization
    """

    def __init__(self, transforms):
        super().__init__(transforms)

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


def random_corner_crop(img, crop_position, size, b_w=0, b_h=0):
    w, h = img.size
    c_h, c_w = size

    if b_w and b_h:
        b_w, b_h = int((w - c_w) * b_w), int((h - c_h) * b_h)

    if c_h > w or c_w > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size, (h, w)))

    if crop_position == 'center':
        return functional.center_crop(img, (c_h, c_w))
    elif crop_position == 'tl':
        return img.crop((b_w, b_h, b_w + c_w, b_h + c_h))
    elif crop_position == 'tr':
        return img.crop((w - c_w - b_w, b_h, w - b_w, b_h + c_h))
    elif crop_position == 'bl':
        return img.crop((b_w, h - c_h - b_h, b_w + c_w, h - b_h))
    elif crop_position == 'br':
        return img.crop((w - c_w - b_w, h - c_h - b_h, w - b_w, h - b_h))
    else:
        raise NotImplementedError


# -------------------------------------------------
# transform re-implementation
# -------------------------------------------------
class ToTensor(torchvision.transforms.ToTensor):

    def randomize_parameters(self):
        pass


class Normalize(torchvision.transforms.Normalize):

    def randomize_parameters(self):
        pass


class Resize(torchvision.transforms.Resize):

    def randomize_parameters(self):
        pass


class CenterCrop(torchvision.transforms.CenterCrop):

    def randomize_parameters(self):
        pass


class ToPILImage(torchvision.transforms.ToPILImage):

    def randomize_parameters(self):
        pass


class RandomCornerCrop(object):
    """Randomly crops the given PIL Image at the four corners or center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        crop_position (sequence): Desired crop position of the crop. If it is None,
            the position is randomly selected. Default choices are
            ('center', 'tl', 'tr', 'bl', 'br').
        crop_scale (sequence): Desired list of scales in the range (0, image_size)
            to randomly crop from corners.
    """
    def __init__(self, size, crop_position=None, crop_scale=1.0, border=0):
        self.size, self.crop_size, self.border = size, size, border
        self.border_w, self.border_h = 0, 0
        self.randomize_corner, self.randomize_scale = True, True
        self.default_positions = ('center', 'tl', 'tr', 'bl', 'br')
        if crop_position is not None:
            self.randomize_corner, self.crop_position = False, crop_position
        if isinstance(crop_scale, tuple):
            self.crop_scale = crop_scale
        elif isinstance(crop_scale, float):
            self.randomize_scale = False
            self.crop_size = self.size * crop_scale
            self.crop_size = (int(self.crop_size), int(self.crop_size))
        else:
            raise Exception('NotDefined')

    def __call__(self, img):
        return random_corner_crop(img, self.crop_position, self.crop_size, self.border_w, self.border_h)

    def randomize_parameters(self):
        if self.randomize_corner:
            self.crop_position = random.choice(self.default_positions)
        if self.randomize_scale:
            if len(self.crop_scale) == 2 and self.crop_scale[0] < 2:
                self.crop_size = self.size * np.random.uniform(low=self.crop_scale[0], high=self.crop_scale[1])
            else:
                self.crop_size = np.random.choice(self.crop_scale)
            self.crop_size = (int(self.crop_size), int(self.crop_size))
        if self.border > 0:
            self.border_w, self.border_h = self.border*np.random.rand(), self.border*np.random.rand()


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p, self.random_value = p, None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if self.random_value < self.p:
            return functional.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

    def randomize_parameters(self):
        self.random_value = random.random()


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.75, 1.0), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.r_scale, self.r_tl = None, None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        w, h = img.size
        r_tl_w, r_tl_h = self.r_tl

        min_length = min(w, h)
        crop_size = int(min_length * self.r_scale)

        i = r_tl_h * (h - crop_size)
        j = r_tl_w * (w - crop_size)

        return functional.resized_crop(img, i, j, crop_size, crop_size, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

    def randomize_parameters(self):
        self.r_scale = random.uniform(*self.scale)
        self.r_tl = [random.random(), random.random()]
