import argparse
import imutils
import time
import dlib
import cv2

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", type=str, required=True,help="path to input image")
# ap.add_argument("-m", "--model", type=str,default="mmod_human_face_detector.dat",help="path to dlib's CNN face detector model")
# ap.add_argument("-u", "--upsample", type=int, default=1,help="# of times to upsample")
# args = vars(ap.parse_args())

# load dlib's CNN face detector
print("[INFO] loading CNN face detector...")
detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
# load the input image from disk, resize it, and convert it from
# BGR to RGB channel ordering (which is what dlib expects)
image = cv2.imread('C:\\Users\\moham\\Pictures\\cv_DP2jpg.jpg')
image = imutils.resize(image, width=600)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# perform face detection using dlib's face detector
start = time.time()
print("[INFO[ performing face detection with dlib...")
results = detector(rgb, 1)
end = time.time()
print("[INFO] face detection took {:.4f} seconds".format(end - start))


# convert the resulting dlib rectangle objects to bounding boxes,
# then ensure the bounding boxes are all within the bounds of the
# input image
boxes = [convert_and_trim_bb(image, r.rect) for r in results]
# loop over the bounding boxes
for (x, y, w, h) in boxes:
	# draw the bounding box on our image
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
# show the output image
cv2.imshow("Output", image)
cv2.imwrite("Output_CNN.jpg", image)
cv2.waitKey(0)