from cv2 import imread, cvtColor, COLOR_BGR2RGB
from final_fns import face_locations, face_encodings
from os import listdir
import pandas as pd

class Encodings:
    def __init__(self, path):
        self.path = path
        self.IMAGE_STORE = []
        self.nameList = listdir(path)    # list of all the images inside the directory
        self.encodedList = []

        for img in self.nameList:  # parsing the images and storing them
            curImg = imread(f'{path}/{img}')
            self.IMAGE_STORE.append(curImg)
        self.findEncodings()

    def __call__(self, *args, **kwargs):
        return self.encodedList

    def findEncodings(self):
        for img in self.IMAGE_STORE:
            img = cvtColor(img, COLOR_BGR2RGB)
            #encode = face_recognition.face_encodings(img)[0]
            camFaceLocs = face_locations(img)
            # similarly here we are giving camFaceLocs, the face location as argument for encodings
            encode = face_encodings(img, camFaceLocs)[0]
            self.encodedList.append(encode)
        print("Encoding COMPLETE")


path = "C:\\Users\\moham\\PycharmProjects\\Major_Project\\DATA"        # directory containing all the students' directories
for path in listdir(path):      # path -> directory containing all the images of the particular student
    Obj = Encodings(path)

    df = pd.DataFrame(Obj())
    df.to_csv(f"{path}\\DataBase.csv")
    print("Encoding Database - 'DataBase.csv' saved")