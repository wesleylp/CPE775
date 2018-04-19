import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import io
from torchvision import utils
import torchvision.transforms.functional as F

import cv2


def show_landmarks(image, landmarks, normalized=False):
    """Show image with landmarks"""

    if isinstance(image, str):
        image = io.imread(image)

    w = h = 1
    if normalized:
        w, h = image.size

    plt.figure()
    plt.imshow(image)

    plt.scatter(landmarks[:, 0]*w, landmarks[:, 1]*h, s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

# Helper function to show a batch
def show_landmarks_batch(sample_batched, normalized=True, nrow=8, padding=2):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch, nrow, padding)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    ncol = np.ceil(batch_size/nrow).astype('int')

    for j in range(ncol):
        for i in range(nrow):
            if i+ j*nrow >= len(landmarks_batch):
                continue

            plt.scatter(landmarks_batch[i+ j*nrow, :, 0].numpy()*im_size + i * im_size + padding*i,
                        landmarks_batch[i+j*nrow, :, 1].numpy()*im_size + j * im_size + padding*j,
                        s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

def get_bbox(x1, y1, x2, y2):
    """ Extracts a square bound box given the coordinates
    """

    x_start = int(np.floor(x1))
    x_end = int(np.ceil(x2))
    y_start = int(np.floor(y1))
    y_end = int(np.ceil(y2))

    width = np.ceil(x_end - x_start)
    height = np.ceil(y_end - y_start)

    # Checks if the box is square
    if width < height:
        diff = height - width
        x_start -=(np.ceil(diff/2.0))
        x_end +=(np.floor(diff/2.0))

    elif width > height:
        diff = width - height
        y_start -=(np.ceil(diff/2.0))
        y_end +=(np.floor(diff/2.0))


    width = x_end - x_start
    height = y_end - y_start
    assert width == height

    rect_init_square = [int(x_start), int(y_start), int(width), int(height)]

    return rect_init_square

def get_bbox_from_landmarks(landmarks):
    x_min = np.min(landmarks[:, 0])
    y_min = np.min(landmarks[:, 1])
    x_max = np.max(landmarks[:, 0])
    y_max = np.max(landmarks[:, 1])

    return get_bbox(x_min, y_min, x_max, y_max)

def enlarge_bbox(bbox, ratio=1.25):
    if ratio <= 1.0:
        return bbox
    
    x_start, y_start, width, height = bbox

    x_end = x_start + width
    y_end = y_start + height

    assert width == height, "width %s is not equal to height %s" %(width, height)
    assert ratio > 1.0 , "ratio is not greater than one. ratio = %s" %(ratio,)

    change = ratio - 1.0

    shift = (change/2.0)*width

    x_start_new = int(np.floor(x_start - shift))
    x_end_new = int(np.ceil(x_end + shift))
    y_start_new = int(np.floor(y_start - shift))
    y_end_new = int(np.ceil(y_end + shift))

    # assertion for increase lenght
    width = int(x_end_new - x_start_new)
    height = int(y_end_new - y_start_new)

    assert height == width

    rect_init_square = [x_start_new, y_start_new, width, height]

    return rect_init_square

def increase_img_size(img, bbox):
        """ this method increases the bounding box size
        if start and end values for the bounding box
        go beyond the image size (from either side)
        and in such a case gets the ratio of padded region
        to the total image size (total_img_size = orig_size + pad)
        """

        x_start, y_start, width_bbox, height_bbox = bbox
        x_end = x_start + width_bbox
        y_end = y_start + height_bbox

        if type(img) == Image.Image:
            width, height = img.size
        else:
            height, width = img.shape[:2]

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

        # checking bounding box size after image padding
        if type(img) == Image.Image:
            expand_img = F.pad(img, (x_extra_start, y_extra_start,
                                 x_extra_end, y_extra_end))
            width, height = expand_img.size
        else:
            expand_img = cv2.copyMakeBorder(img,y_extra_start,y_extra_end,
                                    x_extra_start,x_extra_end,
                                    cv2.BORDER_CONSTANT,value=[0,0,0])
            height, width = expand_img.shape[:2]

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

def rect_to_tuple(rect, padding=0):
    return (int(rect.left()) - padding,
            int(rect.top()) - padding,
            int(rect.right()) + padding,
            int(rect.bottom() + padding))

def padding_rect(rect, padding):
    if type(padding) == int and len(padding) == 1:
        padding = (padding, padding)
    if len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])

    return (rect[0] - padding, rect[1] - padding, rect[2] + padding, rect[3] + padding)
