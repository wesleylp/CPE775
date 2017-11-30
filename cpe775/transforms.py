import numpy as np
import torch
import cpe775.utils.img_utils as utils

from PIL import Image

from torchvision.transforms import transforms
import torchvision.transforms.functional as F


class CropFace(object):
    def __init__(self, size=256, enlarge_ratio=1.7):
        self._enlarge_ratio = enlarge_ratio
        self._size = size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # Extracting the bbox
        bbox = utils.get_bbox_from_landmarks(landmarks)

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
        image, rect, x_extra_start, y_extra_start = self._increase_img_size(image, rect)

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

    def _increase_img_size(self, img, bbox):
        """ this method increases the bounding box size
        if start and end values for the bounding box
        go beyond the image size (from either side)
        and in such a case gets the ratio of padded region
        to the total image size (total_img_size = orig_size + pad)
        """

        x_start, y_start, width_bbox, height_bbox = bbox
        x_end = x_start + width_bbox
        y_end = y_start + height_bbox

        width, height = img.size

        x_extra_start = 0
        x_extra_end = 0
        y_extra_start = 0
        y_extra_end = 0

        if x_start < 0:
            x_extra_start = - x_start
        if x_end > width:
            x_extra_end = x_end - width
        if y_start < 0:
            y_extra_start = - y_start
        if y_end > height:
            y_extra_end = y_end - height

        expand_img = F.pad(img, (x_extra_start, y_extra_start,
                                 x_extra_end, y_extra_end))

        # checking bounding box size after image padding
        width, height = expand_img.size

        if x_extra_start:
            x_start = 0
            x_end += x_extra_start
        if y_extra_start:
            y_start = 0
            y_end += y_extra_start
        if x_extra_end:
            x_end = width
        if y_extra_end:
            y_end = height

        bbox_width = x_end - x_start
        bbox_height = y_end - y_start

        assert bbox_width == bbox_height

        expanded_rect = [x_start, y_start, bbox_width, bbox_height]

        return expand_img, expanded_rect, x_extra_start, y_extra_start

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

        if self.padding > 0:
            image = F.pad(image, self.padding)

            left = top = right = bottom = 0


            if len(self.padding) == 2:
                left, top = self.padding
                right = left
                bottom = left
            elif len(self.padding) == 4:
                left, top, right, bottom = self.padding

            landmarks[:, 0] += left
            landmarks[:, 1] += top

        i, j, h, w = self.get_params(image, self.size)

        landmarks = landmarks - [j/w, i/h]

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
