import os

import matplotlib.pyplot as plt
import pandas as pd
from skimage import io


def read_pts(paths, common_path=''):
    if type(paths) not in (list, set):
        paths = [paths]
    dataset = []

    for path in paths:
        with open(path, 'r') as f:
            # skip the version
            f.readline()
            # reading the number of points
            n_points = int(f.readline().split(':')[1].strip())

            keypoints = []
            for line in f.readlines():
                if line.strip() in ('{', '}'):
                    continue

                keypoints.extend([float(l.strip()) for l in line.split()])
            assert len(keypoints) / 2 == n_points, '{} says {} points, but {} found'.format(
                path, n_points, len(keypoints))
            assert len(keypoints) / 2 == n_points, '{} says {} points, but {} found'.format(
                path, n_points, len(keypoints))

            image_path = os.path.splitext(path)[0] + '.jpg'
            if not os.path.isfile(image_path):
                image_path = os.path.splitext(path)[0] + '.png'

                if not os.path.isfile(image_path):
                    raise ValueError('Could not find the image related to {}'.format(path))

            image_path = os.path.relpath(image_path, common_path)

            dataset.append([image_path] + keypoints)

    return pd.DataFrame(dataset)


def show_landmarks(image, landmarks):
    """Show image with landmarks"""

    if isinstance(image, str):
        image = io.imread(image)
        # image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

    plt.imshow(image)

    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
