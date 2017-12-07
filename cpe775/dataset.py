from torch.utils.data import Dataset

from torchvision.datasets.folder import default_loader

import pandas as pd
import numpy as np
import h5py
import os

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, loader=default_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            loader (callable, optional): A function to load an image given its path.
        """
        self.data_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_df.ix[idx, 0])
        image = self.loader(img_name)

        landmarks = self.data_df.ix[idx, 1:].as_matrix().astype('float32')
        landmarks = landmarks.reshape(-1, 2)

        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class CroppedFaceLandmarksDataset(Dataset):
    """Cropped Face Landmarks dataset.
    Loads images and landmarks from a .npz file with already cropped images
    """

    def __init__(self, data_file, transform=None):
        """
        Args:
            data_file (string): Path to npz file or hdf5 file with the images and landmarks.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        ext = os.path.splitext(data_file)[-1]
        if ext == '.npz':
            data = np.load(data_file)
        elif ext in ('.h5', '.hdf5'):
            data = h5py.File(data_file, 'r')
        else:
            raise ValueError('ext {} not supported'.format(ext))

        self.images, self.landmarks = data['images'], data['landmarks']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        landmarks = self.landmarks[idx]

        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
