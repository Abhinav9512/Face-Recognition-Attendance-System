import cv2
import numpy as np
import winsound
import time
import face_recognition
import os
from datetime import datetime


path = 'imagesAttendence'
images = []
classNames = []
List = os.listdir(path)
for cls in List:
    Img = cv2.imread(f'{path}/{cls}')
    images.append(Img)
    classNames.append(os.path.splitext(cls)[0])

def Encodings(images):
    encodeList = []
    for img in images:

        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as FILE:
        DataList = FILE.readlines()
        nameList = []
        for line in DataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            String = now.strftime(' %d:%b:%y, %H:%M:%S')
            FILE.writelines(f'\n{name},{String}')


encodeListKnown = Encodings(images)
print('Detection Complete')
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.60:
            name = classNames[matchIndex]
            markAttendance(name)
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 128), 2)
            cv2.rectangle(img, (x1, y2 - 40), (x2, y2), (0, 255, 128), cv2.FILLED)
            cv2.putText(img, name, (x1 + 10, y2 - 10), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        else:
            name = 'Unknown'
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 40), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 10, y2 - 10), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)

            time.sleep(2)
            duration = 900
            frequency = 1000
            array = ['you', 'are', 'unknown', 'person']
            i = 0
            n = len(array)
            while i < n:
                winsound.Beep(frequency, duration)
                frequency += 120
                i = i + 1

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)