import random

import numpy as np
import torch
from PIL import Image

import cpe775.utils.img_utils as utils
import torchvision.transforms.functional as F
from torchvision.transforms import transforms

class CropFace(object):
    def __init__(self, size=256, enlarge_ratio=1.7):
        self._enlarge_ratio = enlarge_ratio
        self._size = size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # Extracting the bbox
        bbox = utils.get_bbox_from_landmarks(landmarks)

        x_start, y_start, width, height = bbox

        if self._enlarge_ratio > 1.0:
            # Expanding bbox, giving a margin
            rect = utils.enlarge_bbox(bbox, ratio=self._enlarge_ratio)

            # Verifying the boundaries
            x_min = np.min(landmarks[:, 0])
            y_min = np.min(landmarks[:, 1])
            x_max = np.max(landmarks[:, 0])
            y_max = np.max(landmarks[:, 1])

            x_start, y_start, width, height = rect
            x_end = x_start + width
            y_end = y_start + height

            assert x_start < x_min, "x_start %s is not smaller than x_min %s" %(x_start, x_min)
            assert x_end > x_max, "x_end %s is not bigger than x_max %s" %(x_end, x_max)
            assert y_start < y_min, "y_start %s is not smaller than y_min %s" %(y_start, y_min)
            assert y_end > y_max, "y_end %s is not bigger than y_max %s" %(y_end, y_max)

            # Padding image in order to fix the bbox
            image, rect, x_extra_start, y_extra_start = utils.increase_img_size(image, rect)

            x_start, y_start, width, height = rect

            x_end = x_start + width
            y_end = y_start + height

            if x_extra_start:
                landmarks[:,0] += x_extra_start
            if y_extra_start:
                landmarks[:,1] += y_extra_start

        # Cropping the image around the bbox
        image = F.crop(image, y_start, x_start, height, width)

        landmarks[:, 0] -= x_start
        landmarks[:, 1] -= y_start

        # Downsampling the images
        image = F.resize(image, self._size)

        # Normalizing landmarks
        landmarks[:, 0] /= width
        landmarks[:, 1] /= height

        return {'image': image, 'landmarks': landmarks}

class CenterCrop(transforms.CenterCrop):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, sample):
        """
        Args:
            sample (dict): image (PIL Image) to be cropped and landmarks to be centered.
        Returns:
            sample (dict)
        """
        image, landmarks = sample['image'], sample['landmarks']

        w, h = image.size
        th, tw = self.size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        landmarks -= [j/w, i/h]
        landmarks *= [w/tw, h/th]

        return {'image': F.center_crop(image, self.size), 'landmarks': landmarks}

class RandomCrop(transforms.RandomCrop):
    """ Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __call__(self, sample):
        """
        Args:
            sample (dict): image (PIL Image) to be cropped and landmarks points to be adjusted
        Returns:
            PIL Image: Cropped image.
        """
        image, landmarks = sample['image'], sample['landmarks']

        orig_w, orig_h = image.size

        if self.padding > 0:
            image = F.pad(image, self.padding)

            left = top = right = bottom = 0

            if type(self.padding) == int:
                left = top = right = bottom = self.padding
            elif len(self.padding) == 2:
                left, top = self.padding
                right = left
                bottom = left
            elif len(self.padding) == 4:
                left, top, right, bottom = self.padding

            landmarks[:, 0] += left/orig_w
            landmarks[:, 1] += top/orig_h

        # i: upper pixel coordinate
        # j: left pixel coordinate
        i, j, h, w = self.get_params(image, self.size)

        landmarks -= [j/orig_w, i/orig_h]

        landmarks *= [orig_w/w, orig_h/h]

        image = F.crop(image, i, j, h, w)

        return {'image': image, 'landmarks': landmarks}

class Resize(transforms.Resize):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        super(Resize, self).__init__(size, interpolation)

    def __call__(self, sample):
        """
        Args:
            samples (dict): containing img (PIL Image) and the landmarks points
        Returns:
            resied dict: rescaled image and landmarks points
        """
        image, landmarks = sample['image'], sample['landmarks']

        return {'image': super(Resize, self).__call__(image), 'landmarks': landmarks}

class ToTensor(object):
    """Convert a ``PIL Imag Image to be converted to tensore`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
            sample (dict): `image`  (PIL Image or numpy.ndarray), is the image to be converted to tensor, and `landmarks` containing an 2D array
        Returns:
            Tensor: Converted image.
        """
        image, landmarks = sample['image'], sample['landmarks']
        return {'image': F.to_tensor(image), 'landmarks': torch.from_numpy(landmarks)}

class ToGray(object):
    """Convert a ``PIL Imag Image to be converted to tensore`` or ``numpy.ndarray`` in RGB scale to Gray Scale.
    """

    def __call__(self, sample):
        """
        Args:
            sample (dict): `image`  (PIL Image or numpy.ndarray), is the image to be converted to tensor, and `landmarks` containing an 2D array
        Returns:
            Tensor: Converted image.
        """
        image, landmarks = sample['image'], sample['landmarks']
        return {'image': image.convert('L'), 'landmarks': landmarks}


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    _horizontal_flip_indexes = np.array([
        [1,2,3,4,5,6,7,8,18,19,20,21,22,37,38,39,40,41,42,32,33,49,50,51,61,62,68,60,59],
        [17,16,15,14,13,12,11,10,27,26,25,24,23,46,45,44,43,48,47,36,35,55,54,53,65,64,66,56,57]
    ]) - 1

    def __call__(self, sample):
        """
        Args:
            sample: a dict containing the `image` (PIL image) and the normalized landmarks between [0, 1].
        Returns:
            dict: Randomly flipped image and landmarks
        """
        image, landmarks = sample['image'], sample['landmarks']

        if random.random() < 0.5:

            if landmarks.max() > 1 or landmarks.min() < 0:
                raise ValueError('landmarks points must be between [0,1]')

            # Horizontal flip of all x coordinates:
            landmarks = landmarks.copy()
            landmarks[..., 0] = landmarks[..., 0]*-1 + 1.

            flipped_landmarks = landmarks.copy()

            flipped_landmarks[..., self._horizontal_flip_indexes[0], :] = landmarks[..., self._horizontal_flip_indexes[1], :]
            flipped_landmarks[..., self._horizontal_flip_indexes[1], :] = landmarks[..., self._horizontal_flip_indexes[0], :]
            return {'image': F.hflip(image), 'landmarks': flipped_landmarks}
        return sample


class ToPILImage(transforms.ToPILImage):
    """Convert a tensor or an ndarray to PIL Image.
    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.
    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
            1. If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
            2. If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
            3. If the input has 1 channel, the ``mode`` is determined by the data type (i,e,
            ``int``, ``float``, ``short``).
    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, sample):
        """
        Args:
            sample (dict): contains image (Tensor or numpy.ndarray) and landmarks (Tensor or numpy.ndarray)
        Returns:
            sample (dict): image converted to PIL Image and landmarks
        """
        image, landmarks = sample['image'], sample['landmarks']
        return {'image': F.to_pil_image(image, self.mode),
                'landmarks': landmarks}
