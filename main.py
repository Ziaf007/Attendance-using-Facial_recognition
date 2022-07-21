from final_fns import Worker, Mail
import os
from pandas import read_csv
import datetime

classNames = []
path = "C:\\Users\\moham\\PycharmProjects\\Major_Project\\DATA"
StudentList = os.listdir(path)
# for student in StudentList:
#     for i in range(len(listdir(student))):
#         classNames.append(os.path.splitext(img)[0])

if __name__=="__main__":
    # List of encodings of known faces
    df = read_csv("DATA\\CS181046\\DataBase.csv", index_col=0)
    df_1 = read_csv("DATA\\CS181045\\DataBase.csv", index_col=0)
    df_2 = read_csv("DATA\\EC181024\\DataBase.csv", index_col=0)
    df_3 = read_csv("DATA\\CS181055\\DataBase.csv", index_col=0)
    for i in range(len(df)):
        classNames.append("CS181046")
    for i in range(len(df_1)):
        classNames.append("CS181045")
    for i in range(len(df_2)):
        classNames.append("EC181024")
    for i in range(len(df_3)):
        classNames.append("CS181055")

    df = df.append(df_1, ignore_index=True)
    df = df.append(df_2, ignore_index=True)
    df = df.append(df_3, ignore_index=True)
    # df = read_csv("StudentEncodingsDatabase.csv")

    # df.drop(columns=["label"], axis = 1, inplace = True)
    encodedListKnown = df.values.tolist()

    Obj = Worker(encodedListKnown)
    student,attendance = Obj.pipeline(classNames)

    if attendance:
        print(f'{student} is present.')
    else:
        print("Absent")

    message = []
    try:
        with open("ATTENDANCE\\Attendance - {}.csv".format(str(datetime.date.today())), 'r') as f:
            for entry in f.readlines():
                message.append(entry)
    except FileNotFoundError:
        print("No attendance File found")
        exit(1)
    obj = Mail("emails.txt", "".join(message), "ylnzpzjjkgdxkqii")
    obj.send()