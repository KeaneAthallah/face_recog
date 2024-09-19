import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = '/home/kenn/face_recog/test'
images = []
classNames = []

myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture('/home/kenn/face_recog/mencurigakan.mp4')

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        # If a match is found and the face distance is less than 0.7
        if matches[matchIndex] and faceDis[matchIndex] > 0.7:
            name = classNames[matchIndex].upper()
            color = (0, 255, 0)  # Green for recognized faces
        else:
            name = "SUSPICIOUS"
            color = (0, 0, 255)  # Red for unrecognized or low-confidence faces

        # Scale face locations back to original frame size
        y1, x1, y2, x2 = faceLoc
        y1, x1, y2, x2 = y1 * 4, x1 * 4, y2 * 4, x2 * 4

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
