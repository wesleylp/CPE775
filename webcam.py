import argparse
import numpy as np
import time

import torch
import dlib
import cv2

from cpe775.align import AlignDlib
from cpe775.networks.openface import OpenFace
from cpe775.face_detector import FaceEmbedding
from cpe775.networks.resnet import resnet18
from cpe775.face_detector import LandmarksDetector
from cpe775.face_detector import FaceClassify

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--record', action='store_true', default=False)
    parser.add_argument('--scale', type=float, default=0.25)
    parser.add_argument('--video-capture', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    # loading the face detector
    face_detector = dlib.get_frontal_face_detector()
    # loading the align class
    align = AlignDlib()
    # loading the embedding model
    embedding = FaceEmbedding('models/openface/nn4.small2.v1.pth',
                              cuda=args.cuda)
    # loading landmarks regressor
    net = resnet18(in_shape=(-1, 1, 224, 224), out_shape=(-1, 68, 2))
    landmarks_detector = LandmarksDetector(net, 'models/landmarks_detector.pth', cuda=args.cuda)
    # loading face classification model
    classify = FaceClassify('models/face_classification_svm.pkl')

    video_capture = cv2.VideoCapture(args.video_capture)

    fps = video_capture.get(cv2.CAP_PROP_FPS)

    print('Frames per second using video.get(cv2.CAP_PROP_FPS: {}'.format(fps))

    # Define the codec and create VideoWriter object
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter('record.mp4',fourcc, 24.0, (640,480))

    # Initialize some variables
    face_locations = []
    face_names = []
    face_landmarks = []
    face_probas = []
    process_this_frame = True
    frame_count = 0
    global_start = time.time()
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=args.scale, fy=args.scale)

        # Only process every other frame of video to save time
        if process_this_frame:
            start = time.time()
            face_locations = []
            face_names = []
            face_probas = []
            face_landmarks = []

            # Find all the faces and face encodings in the current frame of video
            detected_faces = face_detector(small_frame, 2)

            # Loop through each face we found in the image
            for i, face_rect in enumerate(detected_faces):

                # Find the landmarks given the image and the bounding boxes
                landmarks, rect = landmarks_detector.from_rect(small_frame, face_rect)

                face_locations.append(rect)

                face_landmarks.append(landmarks)

                # Align and resize the face given the landmarks
                thumbnail = align(small_frame, landmarks)

                # Extract the embedding from the aligned face
                embed = embedding(thumbnail)

                # Classify
                name, prob = classify(embed)

                face_probas.append(prob)

                if prob > 0.5:
                    face_names.append(name)
                else:
                    face_names.append('unknown')

        process_this_frame = not process_this_frame


        # Display the results
        for (left, top, right, bottom), name, prob, landmarks in zip(face_locations, face_names, face_probas, face_landmarks):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top = int(top/args.scale) - 15
            right = int(right/args.scale) + 15
            bottom = int(bottom/args.scale) + 45
            left = int(left/args.scale) - 15

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, '{}, {:.2f}%'.format(name, prob*100), (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
            for (x, y) in landmarks:
                cv2.circle(frame, (int(x/args.scale), int(y/args.scale)), 1, (0, 0, 255), -1)
        
            if args.verbose:
                print('{}'.format(time.time()-start))
        if args.record:
            out.write(frame)
        # Display the resulting image
        cv2.imshow('Video', frame)
 
        frame_count += 1

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Estimated fps: {}'.format(frame_count/(time.time()-global_start)))
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
