import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
import sys
import face_recognition
import imutils.face_utils
from imutils import face_utils

#import imutils.face_utils
#image read
imq = cv2.imread('C:\\Users\\moham\\Pictures\\cv_DP2jpg.jpg', 0)
#image show
cv2.imshow('image', img)
#image show without hanging up
cv2.waitKey(0)
#converting dtype of numpy.ndarray into float(32 bit) /255 ????
im = np.float32(imq)/255.0
#depth command ????
depth = cv2.CV_32F

#Calculating gradient
#The Sobel Operator is a discrete differentiation operator.
#It computes an approximation of the gradient of an image intensity function.
gx = cv2.Sobel(im, depth, 1, 0, ksize=1)
gy = cv2.Sobel(im, depth, 0, 1, ksize=1)
#computing the magnitude and angle of the gradient(2D vectors)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

#plotting the picture
plt.figure(figsize=(12,8))
plt.imshow(mag)
plt.show()

#face detection
# HOG + Linear SVM = dlib.get_frontal_face_detector()
#MMOD CNN = dlib.cnn_face_detection_model_v1(modelPath)
#WE are using HOG

#The get_frontal_face_detector function does not accept any parameters. A call to it returns the pre-trained HOG +
#Linear SVM face detector included in the dlib library.
#Dlib’s HOG + Linear SVM face detector is fast and efficient. By nature of how the Histogram of Oriented Gradients (HOG)
#descriptor works, it is not invariant to changes in rotation and viewing angle.

#For more robust face detection, you can use the MMOD CNN face detector, available via the cnn_face_detection_model_v1 function
#this method accepts a single parameter, modelPath, which is the path to the pre-trained MMOD-Model_FaceDetection.dat
#file residing on disk.

face_detect = dlib.get_frontal_face_detector()
rects = face_detect(imq, 1) #gettinf the coordinates for the rectangle on the front face
for (i, rect) in enumerate(rects): #actually getting the coordinates from their tuple format
    (x, y, w, h) = face_utils.rect_to_bb(rect)
cv2.rectangle(imq, (x, y), (x + w, y + h), (0, 255, 128), 3)  #plotting the rect over the image

plt.figure(figsize=(12, 8))
plt.imshow(imq, cmap='gray')  #cmap is colormap instance
plt.show()

# RdBu_r, RdGy  RdGy_r RdYlBu_r RdYlBu