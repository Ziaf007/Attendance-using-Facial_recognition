import pandas as pd
import numpy as np
import cv2
import dlib
from imutils import face_utils
import face_recognition
import os
import time

start_time = 0
vid_cap = cv2.VideoCapture("WhatsApp Video 2021-06-07 at 2.42.33 PM.mp4")       # creating a frame capturing object

# ID = input("Enter your enrol No. (in Uppercase letters): ")
ID = "CS181055"

if not os.path.exists(ID):          # creating a directory to store all the sample images
    os.mkdir(ID)

if not vid_cap.isOpened():          # checking if webcam is open or not
    print("WebCam not detected. Sorry !.")
    vid_cap.release()
    exit()

data = []   # a list to store [encodings, label]

def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()

frame_count = 0   # counter to detect only 30 samples per person
while True and frame_count < 31:

    time_elapsed = time.time() - start_time
    ret, frame = vid_cap.read()         # ret is a bool value, and frame is a single frame of the video.
    if not ret:                         # when the capture ends
        print("Cannot capture any more frames. Perhaps the video may have ended")
        break

    if time_elapsed < (1./45):
        continue

    cv2.imshow("you", frame)
    k = cv2.waitKey(0)

    if k == ord('q'):
        print("quitting")
        break
    elif k == ord('f'):
        print("Capturing")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converting to GrayScale
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # converting to RGB for face_recognition


        # OPEN THIS COMMENT SECTION TO CAPTURE IMAGES WITH BOX BOUNDED FACES
        detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")     # the actual detector model (HOG) object
        # rects = detector(rgb, 1)
        rects = [_trim_css_to_bounds(_rect_to_css(face.rect), rgb.shape) for face in detector(rgb, 1)]
        # passing the image to the model object to obtain tuples

        try:
            (y,w,h,x) = rects[0]
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 128), 3)  # plotting the rect over the image
        except IndexError:
            print("Face Not aligned try again !!!")
            continue

        detected_boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model='cnn')   # box co-ordinates
        encoding = face_recognition.face_encodings(rgb, detected_boxes)     # encodings of this particular sample
        frame_count += 1
        # print(encoding[0])
        # data.append({"encoding": encoding, "label": ID})        # appending to the final data list
        data.append(encoding[0])
        # print(data)
        # input()


        # cv2.imshow(f'frame:{frame_count}', frame)  # for debugging purposes use along with next lines
        # cv2.waitKey(0)
        # cv2.destroyWindow(f'frame:{frame_count}')
        start_time = time.time()

        # print("Capturing Images now, look at the webcam !!!")
        # cv2.imwrite(f"{ID}\\me_{frame_count}.jpg", gray)
        # if cv2.waitKey(1) == ord('q'):
        #     break

    elif k == ord("e"):
        print("Jumping to next frame")
        start_time = time.time()
        continue

    else:
        raise Warning("Please hit a valid key: \"q\", \"f\" or \"e\".")

print("\n\nCapture complete")
DF = pd.DataFrame(data=[d for d in data], dtype=np.float64)
# converting the stored list into a dataframe in the format samples x 128
DF.to_csv(f"{ID}\\DataBase.csv", line_terminator="")          # exporting the dataframe for training purpose.
vid_cap.release()                                # releasing memory from video capture object
cv2.destroyAllWindows()                          # destroying all the open windows