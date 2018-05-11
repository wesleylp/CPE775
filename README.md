# Real-Time Deep-Learning-Based System for Facial Recognition

We designed a system for facial recognition using machine learning and computer vision techniques.
This work was developed, with major contributions of [Igor M. Quintanilha](https://github.com/igormq), through CPE775 - Machine Learning class taught at Federal University of Rio de Janeiro, Brazil.

The system is deep-learning-based and is composed of five main steps: face segmentation, facial features detection, face alignment, embedding, and classification. 
Deep learning methods is employed for fiducial points extraction and embedding.
For classification, we use Support Vector Machine (SVM) is since it is fast for both training and inference.
The system achieves an error rate of 0.12103 for facial features detection and 0.05 for facial recognition.
Besides, it is capable to run in real-time using a webcam.

## The System

### Face Detection
This work uses a Histogram of Oriented Gradient (HOG) based method, the Max-Margin Object Detection (MMOD), implemented by using [dlib](http://dlib.net/) library. 
The segmented face is delivered to the facial feature (landmarks) extraction step.

### Landmarks Extraction
We use ResNet-18 architecture for being employed in many state-of-the-art computer vision algorithms due to its simplicity and high generalization capability.

### Face Alignment
We apply an affine transformation to align the faces from the image in such way that the nose, eyes, and mouth are aligned with the center of the image as better as possible. 
To do so, we use two functions from OpenCV library: `getAffineTransform`, that returns the rotation and translation necessary to take the original points to the desired ones (an average mask calculated from the points in the training set); and the `warpAffine`, that applies the transformation, which also scales the resulting image.

### Embedding
A network is trained to minimize the so-called
[triplet loss](https://ieeexplore.ieee.org/document/7298682/) – at each iteration, the network is fed with three images: two distinct images of the same person and an image of a different person.

### Classification
The last step is greatly simplified.
We apply an SVM is applied to classify each vector as belonging to a person or not.
The SVM was chosen since it is fast for both training and inference.

----
## Installation

### Requirements
* Linux (might work with other Operational Systems, but it was not tested).
* [Conda](https://conda.io/docs/user-guide/install/index.html): package, dependency and environment management.

### Guide

1. Download the source code 
([`.zip`](https://github.com/wesleylp/CPE775/archive/v1.0.zip)
or
[`.tar.gz`](https://github.com/wesleylp/CPE775/archive/v1.0.tar.gz))
and extract it;
2. Open the terminal, go to the folder where you extracted the code and run the following command `conda env create -f requirements.yml`
to generate the `face` environment
3. Activate your environment by typing `source activate face`.
To deactivate the environment type in `source deactivate`.
NOTE: Remember to always activate the environment before running the code.
4. Download the [pre-trained models](https://github.com/wesleylp/CPE775/releases/download/v1.0/models.tgz) and extract them to the `models` folder;
5. Follow the instructions in the [`notebooks`](https://github.com/wesleylp/CPE775/tree/master/notebooks) folder.

----
## Usage
To use the real-time application:

1. Crop 20 images of the person that will be recognized by doing [FILL THIS OUT...]
2. Train the pre-trained SVM model by executing [FILL THIS OUT...]
3. Run the application [FILL THIS OUT...]

----
## Change log
* 21-Dez-2017: Launch (class presentation)

----
## Thanks
* [Igor](https://github.com/igormq) for being the major contributor to this project;
* All authors whose works were used as a reference.
