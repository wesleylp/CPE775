'''
    Find faces using HOG;
    code based on https://gist.github.com/ageitgey/1c1cb1c60ace321868f7410d48c228e1
'''

import argparse

import dlib
from skimage import io


if __name__ == '__main__':
    # You can download the required pre-trained face detection model here:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    predictor_model = "models/shape_predictor_68_face_landmarks.dat"

    # Argument parser
    parser = argparse.ArgumentParser(description='Find faces')
    parser.add_argument('file_name', metavar='FILE',
                        help='Path to an image')

    args = parser.parse_args()

    image = io.imread(args.file_name)

    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)

    # Run the HOG face detector on the image data.
    # The result will be the bounding boxes of the faces in our image.
    detected_faces = face_detector(image, 1)

    print("I found {} faces in the file {}".format(len(detected_faces), args.file_name))

    win = dlib.image_window()

    # Open a window on the desktop showing the image
    win.set_image(image)

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):

        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

        # Draw a box around each face we found
        win.add_overlay(face_rect)

        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image, face_rect)

        # Draw the face landmarks on the screen.
        win.add_overlay(pose_landmarks)

    # Wait until the user hits <enter> to close the window
    dlib.hit_enter_to_continue()
