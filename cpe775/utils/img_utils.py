import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from torchvision import utils


def show_landmarks(image, landmarks):
    """Show image with landmarks"""

    if isinstance(image, str):
        image = io.imread(image)

    plt.figure()
    plt.imshow(image)

    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
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
