import numpy as np
import torch
from sklearn.externals import joblib

import cv2
import cpe775.utils.img_utils as utils

from .networks.openface import OpenFace


class Base(object):

    def __init__(self, net, model, size, cuda=False):
        self.net = net
        self.model = model
        self.cuda = cuda
        self.size = size

        self.net.load_state_dict(torch.load(self.model)['state_dict'])
        self.net.eval()

        if self.cuda:
            self.net.cuda()

    def __call__(self, x, bgr=True):

        if bgr and x.shape[-1] == 3:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        x = cv2.resize(x, (self.size, self.size),
                       interpolation = cv2.INTER_CUBIC)

        ndims = 4
        if len(x.shape) == 2:
            x = x[np.newaxis, ..., np.newaxis]
            ndims = 2
        elif len(x.shape) == 3: # W, H, C
            x = x[np.newaxis, ...]
            ndims = 3

        # tranpose channel
        x = x.transpose((0, 3, 1, 2))

        if x.dtype == np.uint8:
            x = x/255

        x_var = torch.autograd.Variable(torch.from_numpy(x).float(),                                            volatile=True)

        y = self.net(x_var).data.cpu().numpy()

        if ndims != 4:
            return y[0, ...]

        return y

class LandmarksDetector(Base):

    def __init__(self, net, model, size=224, cuda=False):
        super(LandmarksDetector, self).__init__(net, model, size, cuda)


    def from_rect(self, image, rect, enlarge_ratio = 1.7, bgr=True, to_gray=True):
        rect = utils.rect_to_tuple(rect)
        bbox = utils.get_bbox(*rect)

        large_bbox = utils.enlarge_bbox(bbox, enlarge_ratio)

        increased_img, large_bbox, _, _ = utils.increase_img_size(image, large_bbox)

        left, top, w, h = large_bbox

        cropped_image = increased_img[top:top + h, left:left + w, :]

        return self(cropped_image, bgr, to_gray) * [w, h] + [left, top], rect

    def __call__(self, x, bgr=True, to_gray=True):
        if to_gray:
            cvtColor = cv2.COLOR_BGR2GRAY if bgr else cv2.COLOR_RGB2GRAY
            x = cv2.cvtColor(x, cvtColor)

        y = super(LandmarksDetector, self).__call__(x, bgr)

        return y

class FaceEmbedding(Base):

    def __init__(self, model, net=OpenFace(), size=96, cuda=False):
        super(FaceEmbedding, self).__init__(net, model, size, cuda)


    def __call__(self, x, bgr=True):
        y = super(FaceEmbedding, self).__call__(x, bgr)

        return y


class FaceClassify(object):

    def __init__(self, model_path):
        self.model, self.le = joblib.load(model_path)

    def __call__(self, x):

        ndims = 2
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
            ndims = 1

        predictions = self.model.predict_proba(x)

        if ndims == 1:
            predictions = predictions[0, :]

        max_idxs = np.argmax(predictions, axis=0)

        return self.le.inverse_transform(max_idxs), predictions[..., max_idxs]
