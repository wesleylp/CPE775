from __future__ import print_function, division
import argparse
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import dlib
from skimage import io
from collections import OrderedDict
import sys

#sklearn
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle

#Data augmentation library
import imgaug as ia
from imgaug import augmenters as iaa

#Custom library
from cpe775.align import *
from cpe775.face_detector import FaceEmbedding, LandmarksDetector
from cpe775.model import Model
from cpe775.networks.openface import OpenFace
from cpe775.networks.resnet import resnet18
from cpe775.utils.img_utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()


    # loading the face detector
    face_detector = dlib.get_frontal_face_detector()
    # loading the align class
    align = AlignDlib()
    # loading landmarks regressor
    net = resnet18(in_shape=(-1, 1, 224, 224), out_shape=(-1, 68, 2))
    landmarks_detector = LandmarksDetector(net,'models/landmarks_detector.pth', cuda=args.cuda)
    # loading openFace model
    ckpt = torch.load('models/openface/nn4.small2.v1.pth')
    openface = OpenFace()
    openface.load_state_dict(ckpt['state_dict'])

    #Getting the people's name from the folders
    people_folder = 'data/pics/'
    people = os.listdir(people_folder)

    #dictionary to track faces not found
    face_not_found = {}

    for person in people:

        print('\nGetting face images from: {}'.format(person))
        file_ = glob.glob(people_folder+person+'/*')
        face_not_found[person] = []

        for f in file_:

            if f[-4:].lower() not in ('.jpg', '.png', 'jpeg'):
                continue

            # Load the image
            image = cv2.imread(f)

            # Run the HOG face detector on the image data
            detected_faces = face_detector(image, 1)

            print('Found {} faces in the image file {}'.format(len(detected_faces), f))

            # not use the image if the number of faces detected is different from 1
            if len(detected_faces) is not 1:
                #saving for future reference
                face_not_found[person].append((len(detected_faces), f))
                continue

            # Detected faces are returned as an object with the coordinates
            # of the top, left, right and bottom edges
            face_rect = next(iter(detected_faces))
            print('- Face found at Left: {} Top: {} Right: {} Bottom: {}'\
                .format(face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

            # Find the landmarks given the image and the bounding boxes
            landmarks, _ = landmarks_detector.from_rect(image, face_rect)

            left_eye = np.mean(landmarks[range(36,42), :], axis=-2)
            right_eye = np.mean(landmarks[range(42,48), :] , axis=-2)

            # Align and resize the face given the landmarks
            #thumbnail = align(image, landmarks * [w, h] + [left, top])
            thumbnail = align(image, landmarks)

            path, name = os.path.split(os.path.abspath(f))
            name, ext = os.path.splitext(name)

            if not os.path.isdir(os.path.join(path, 'cropped')):
                os.makedirs(os.path.join(path, 'cropped'))

            cv2.imwrite(os.path.join(path, 'cropped', '{}{}'.format(name, ext)), thumbnail)

    #TODO: better face the problem when the algorithm don't find only 1 face
    if any(face_not_found):
        print('\nuh-oh! We found some problems...')
        for k,v in face_not_found.items():
            print('\n{}:'.format(k))
            for nb_faces, path in v:
                print('found {} faces in file: {}'.format(nb_faces, path))

        raise ValueError('Try replacing these images')
    else:
        print('All images are good!')

    #loading faces
    #TODO: better way to choose number of images n
    #it's hard coded
    n = 10 #number of images of each person
    files = [str(f) for person in people \
            for f in np.random.choice(glob.glob(people_folder+'{}/cropped/*'\
                                                .format(person)), n, replace=False)]

    X = np.stack([io.imread(f) for f in files], axis=0)/255
    y = np.hstack([n*[person] for person in people])
    le = LabelEncoder()
    y = le.fit_transform(y)

    #Shuffling samples (here we use all image as training samples)
    X_train, y_train = shuffle(X, y, random_state=42)

    #Transform input
    X_train_var = torch.autograd.Variable(torch.from_numpy(X_train.transpose((0, 3, 1, 2))), volatile=True).float()

    # Embedding and convert to numpy
    X_train_npy = openface(X_train_var).data.cpu().numpy()

    # Training an SVM classifier
    model = SVC(probability=True, C=2, kernel='linear', tol=1e-6)
    print('\nRegistering users... ')
    model.fit(X_train_npy, y_train)

    #saving the model
    from sklearn.externals import joblib
    joblib.dump((model, le), 'models/face_classification_svm-application.pkl')
    print('\nDone!\n')