import dlib
from PIL import ImageFile
import pickle
from cv2 import VideoCapture,CAP_DSHOW, resize, cvtColor, rectangle, putText, imshow, waitKey, COLOR_BGR2RGB, FONT_HERSHEY_SIMPLEX, destroyAllWindows
from datetime import datetime, date
import numpy as np
import os
import smtplib, ssl

try:
    import face_recognition_models
except Exception:
    print("Please install `face_recognition_models` with this command before using this:\n")
    print("pip install git+https://github.com/ageitgey/face_recognition_models")
    quit()

class Libs:
    def __init__(self):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.face_detector = dlib.get_frontal_face_detector()

        self.predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
        self.pose_predictor_68_point = dlib.shape_predictor(self.predictor_68_point_model)

        self.predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
        self.pose_predictor_5_point = dlib.shape_predictor(self.predictor_5_point_model)

        self.cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(self.cnn_face_detection_model)

        self.face_recognition_model = face_recognition_models.face_recognition_model_location()
        self.face_encoder = dlib.face_recognition_model_v1(self.face_recognition_model)

        self.today = date.today()


    def _rect_to_css(self, rect):
        """
        Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
        :param rect: a dlib 'rect' object
        :return: a plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return rect.top(), rect.right(), rect.bottom(), rect.left()


    def _css_to_rect(self, css):
        """
        Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :return: a dlib `rect` object
        """
        return dlib.rectangle(css[3], css[0], css[1], css[2])


    def _trim_css_to_bounds(self, css, image_shape):
        """
        Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.
        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :param image_shape: numpy shape of the image array
        :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


    def face_distance(self, face_encodings, face_to_compare):
        """
        Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        for each comparison face. The distance tells you how similar the faces are.
        :param face_encodings: List of face encodings to compare
        :param face_to_compare: A face encoding to compare against
        :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
        """
        if len(face_encodings) == 0:
            return np.empty((0))

        # np.array(face_encodings, dtype=float)
        # np.array(face_to_compare, dtype=float)
        return np.linalg.norm(face_encodings - face_to_compare, axis=1)



    def _raw_face_locations(self, img, number_of_times_to_upsample=1, model="hog"):
        """
        Returns an array of bounding boxes of human faces in a image
        :param img: An image (as a numpy array)
        :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
        :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                      deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
        :return: A list of dlib 'rect' objects of found face locations
        """
        if model == "cnn":
            return self.cnn_face_detector(img, number_of_times_to_upsample)
        else:
            return self.face_detector(img, number_of_times_to_upsample)


    def face_locations(self, img, number_of_times_to_upsample=1, model="hog"):
        """
        Returns an array of bounding boxes of human faces in a image
        :param img: An image (as a numpy array)
        :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
        :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                      deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
        :return: A list of tuples of found face locations in css (top, right, bottom, left) order
        """
        if model == "cnn":
            return [self._trim_css_to_bounds(self._rect_to_css(face.rect), img.shape) for face in self._raw_face_locations(img, number_of_times_to_upsample, "cnn")]

        else:
            return [self._trim_css_to_bounds(self._rect_to_css(face), img.shape) for face in self._raw_face_locations(img, number_of_times_to_upsample, model)]

    def _raw_face_locations_batched(self, images, number_of_times_to_upsample=1, batch_size=128):
        """
        Returns an 2d array of dlib rects of human faces in a image using the cnn face detector
        :param images: A list of images (each as a numpy array)
            :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
        :return: A list of dlib 'rect' objects of found face locations
        """
        return self.cnn_face_detector(images, number_of_times_to_upsample, batch_size=batch_size)


    def batch_face_locations(self, images, number_of_times_to_upsample=1, batch_size=128):
        """
        Returns an 2d array of bounding boxes of human faces in a image using the cnn face detector
        If you are using a GPU, this can give you much faster results since the GPU
        can process batches of images at once. If you aren't using a GPU, you don't need this function.
        :param images: A list of images (each as a numpy array)
        :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
        :param batch_size: How many images to include in each GPU processing batch.
        :return: A list of tuples of found face locations in css (top, right, bottom, left) order
        """
        def convert_cnn_detections_to_css(self, detections):
            return [self._trim_css_to_bounds(self._rect_to_css(face.rect), images[0].shape) for face in detections]

        raw_detections_batched = self._raw_face_locations_batched(images, number_of_times_to_upsample, batch_size)

        return list(map(convert_cnn_detections_to_css, raw_detections_batched))



    def _raw_face_landmarks(self, face_image, face_locations=None, model="large"):
        if face_locations is None:
            face_locations = self._raw_face_locations(face_image)
        else:
            face_locations = [self._css_to_rect(face_location) for face_location in face_locations]

        pose_predictor = self.pose_predictor_68_point

        if model == "small":
            pose_predictor = self.pose_predictor_5_point

        return [pose_predictor(face_image, face_location) for face_location in face_locations]


    def face_landmarks(self, face_image, face_locations=None, model="large"):
        """
        Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image
        :param face_image: image to search
        :param face_locations: Optionally provide a list of face locations to check.
        :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
        :return: A list of dicts of face feature locations (eyes, nose, etc)
        """
        landmarks = self._raw_face_landmarks(face_image, face_locations, model)
        landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

        # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
        if model == 'large':
            return [{
                "chin": points[0:17],
                "left_eyebrow": points[17:22],
                "right_eyebrow": points[22:27],
                "nose_bridge": points[27:31],
                "nose_tip": points[31:36],
                "left_eye": points[36:42],
                "right_eye": points[42:48],
                "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
                "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
            } for points in landmarks_as_tuples]
        elif model == 'small':
            return [{
                "nose_tip": [points[4]],
                "left_eye": points[2:4],
                "right_eye": points[0:2],
            } for points in landmarks_as_tuples]
        else:
            raise ValueError("Invalid landmarks model type. Supported models are ['small', 'large'].")


    def face_encodings(self, face_image, known_face_locations=None, num_jitters=1, model="small"):
        """
        Given an image, return the 128-dimension face encoding for each face in the image.
        :param face_image: The image that contains one or more faces
        :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
        :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
        :param model: Optional - which model to use. "large" or "small" (default) which only returns 5 points but is faster.
        :return: A list of 128-dimensional face encodings (one for each face in the image)
        """
        raw_landmarks = self._raw_face_landmarks(face_image, known_face_locations, model)
        return [np.array(self.face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


    def compare_faces(self, known_face_encodings, face_encoding_to_check, tolerance=0.42):
        """
        Compare a list of face encodings against a candidate encoding to see if they match.
        :param known_face_encodings: A list of known face encodings
        :param face_encoding_to_check: A single face encoding to compare against the list
        :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
        :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
        """
        return list(self.face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


    def markAttendance(self, name):
        """
            Given a name, marks its attendance in the database.
            :param name: name of the face recognised whose attendance has to be marked
            :return: An attendance update in data base
        """

        # try:
        #     with open(f'Attendance - {str(self.today)}.csv', 'r') as db:
        #         pass
        # except FileNotFoundError:
        # f = open(f'Attendance - {str(self.today)}.csv', 'w')
        # f.close()
        # finally:
        if not os.path.exists("ATTENDANCE"):
            os.mkdir("ATTENDANCE")

        with open(f'ATTENDANCE\\Attendance - {str(self.today)}.csv', 'w+') as db:
            myDataList = db.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in set(nameList):
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                myDataList.append(f'\n{name},{dtString}')
                print("Attendance Marked")
            db.writelines(myDataList)
            return True

class Worker(Libs):
    name_dict = {0: "Lakshya", 1: "EC181024", 2: "CS181046", 3: "CS181045", 4: "CS181055"}
    cap = VideoCapture(0)

    def __init__(self, encodedListKnown, output_size=(0,0), resize_scale=0.25, font=FONT_HERSHEY_SIMPLEX):
        super().__init__()
        self.capture = Worker.cap
        self.Xscale = resize_scale
        self.Yscale = resize_scale
        self.output_size = output_size
        self.font = font
        self.known_encodings = encodedListKnown
        self.OP_Status = False

    def Draw(self, frame, faceLoc, name):
        """"
            Method to draw the bounding box around the face in cam feed
            :param frame: the frame on which the box is to be drawn
            :param faceLoc: the coordinates of the bounding box
            :param name: the name which needs to be drawn on the box
            :return: None
        """
        # to show box in webcam with name
        y1, x2, y2, x1 = faceLoc
        # multiplying the location coordinates by 4 bcz be derived them for scaled down image
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # addition box for name space
        # rectangle(frame, (x1, y1 - 35), (x2, y2), (0, 255, 0))
        # name as text
        putText(frame, name, (x1 + 6, y2 - 6), self.font, 1, (100, 45, 240), 2)

    def SVM(self,FaceEncodings, FaceLocations, frame ):
        model = pickle.load(open("model.pkl", "rb"))  # unpickling the trained model
        for encodeFace, faceLoc in zip(FaceEncodings, FaceLocations):
            predict_ = model.predict(FaceEncodings)  # outputs Label of the face recognised
            name = Worker.name_dict.get(predict_[FaceLocations.index(faceLoc)])  # retreiving the enrollment no.
            # using the label

            self.Draw(frame, faceLoc, name)  # Drawing the bounding box on the frame
            for name in predict_:  # each face recognised in the frame
                SUCCESS = self.markAttendance(name_dict.get(name))  # mark the attendance and store the
                                                                    # operation status in SUCCESS
                # self.markAttendance(name_dict.get(name))
                return SUCCESS

    def pipeline(self, classNames, method="euclidean", RunTime=3):
        """
                    Driver pipeline to perform the recognition and mark attendance
                    :param method: method used to recognise the face SVM or euclidean (default: "euclidean")
                    :param RunTime: max no. of minutes the script runs to capture the face.
                    :return: None
        """
        timer = int(datetime.now().strftime("%M"))

        while True and timer-int(datetime.now().strftime("%M"))<RunTime and not self.OP_Status:   #video is a just a large no. of images
            success, frame = self.capture.read()
            waitKey(1)

            if success:
                # reducing the size of image bcz its real time
                if any(self.output_size):
                    imgSmall = resize(frame, dsize=self.output_size, dst=None, fx=0, fy=0, interpolation=None)
                else:
                    imgSmall = resize(frame, dsize= (0,0), dst=None, fx=self.Xscale, fy=self.Yscale, interpolation=None)

                imgSmall = cvtColor(imgSmall,COLOR_BGR2RGB)  # converting to RGB because face_recognition works with that only

                # finding the encoding of the image from the webcam:-

                # there might be multiple faces so we removed the [0] after it as then it would have only taken the first location
                camFaceLocs = self.face_locations(imgSmall)
                # input()
                # similarly here we are giving camFaceLocs, the face location as argument for encodings
                camFaceEncodings = self.face_encodings(imgSmall, camFaceLocs)

                if method == "euclidean":
                    # finding the matches
                    # first we will iterate through all the faces found in the cam
                    for encodeFace, faceLoc in zip(camFaceEncodings, camFaceLocs):
                        matches = self.compare_faces(self.known_encodings, encodeFace)
                        faceDist = self.face_distance(self.known_encodings, encodeFace)
                        print("Comparing now")
                        # it will give us the values for all the faces from the list, lowest distance will be the best match
                        # print(faceDist)
                        matchIndex = np.argmin(faceDist)

                        if matches[matchIndex]:     # if a match is found perform:
                            name = classNames[matchIndex]   # finding the name of the match
                            self.Draw(frame, faceLoc, name)     # Drawing the box around the face
                            # calling the func
                            self.OP_Status = self.markAttendance(name)       # marking attendance and storing the operation status in SUCCESS
                            # self.markAttendance(name)                 # alternative for demo purposes. continues live stream capture

                elif method == "svm":           # This technique is only useful for closed datasets.
                                                # i.e. it cannot detect unknown persons.
                                                # Every unknown face will be classified as one of the samples

                    self.OP_Status = self.SVM(camFaceEncodings, camFaceLocs, frame)

                # imshow("you", frame)    # Displays the drawn livestream
                # if waitKey(1) & 0xFF == ord('q'):
                #     break

            elif not success:     # no frame was captured.
                print("No frame Detected, perhaps the video stream has ended.")
                self.capture.release()
                destroyAllWindows()
                exit()

        destroyAllWindows()
        return name,self.OP_Status

class Mail():
    def __init__(self, database, message, passphrase, encryption_protocol="TLS"):
        self.DB = database
        self.sender = "mohammadfaiz007.mf@gmail.com"
        self.receiver = ""
        self.domain = "gmail.com"
        self.host = "smtp.{}".format(self.domain)
        self.username = self.sender
        self.passphrase = passphrase
        self.SMTPobj = smtplib.SMTP(self.host,587)
        self.encryption_protocol = encryption_protocol
        self.message = message

        self.context = ssl.create_default_context()

    def send(self):
        def decrypt(string):
            decryptedE = ""
            for letter in string.strip():
                if letter == ' ':
                    decryptedE += ' '
                else:
                    decryptedE += chr(ord(letter) - 5)
            return decryptedE

        with open(self.DB, 'r') as DB:
            Senders_list = DB.readlines()
            for email in Senders_list:
                self.receiver=decrypt(email)
                if self.encryption_protocol == "TLS":
                    self.port = 587
                    self.server = smtplib.SMTP(self.host, self.port)
                    self.server.starttls(context=self.context)  # Secure the connection
                    self.server.login(self.username, self.passphrase)
                    self.server.sendmail(self.sender, self.receiver, self.message)

                elif self.encryption_protocol == "SSL":
                    self.port = 465
                    with smtplib.SMTP_SSL(self.host, self.port, context=self.context) as self.server:
                        server.login(self.username, self.passphrase)
                        server.sendmail(self.sender, self.receiver, self.message)

                else:
                    raise ("Invalid Encryption protocol.\n Please select either \"TLS\" or \"SSL\"")