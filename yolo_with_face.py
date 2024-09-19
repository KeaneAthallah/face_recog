import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
from telegram import Bot
from telegram.ext import Application
import asyncio
import torch
import math
from ultralytics import YOLO

# Telegram Bot setup
BOT_TOKEN = '7270440999:AAGHWa8CdB3oL5kR3iwMsj0QgfSyi77RiXU'
CHAT_ID = '7513725643'

application = Application.builder().token(BOT_TOKEN).build()

# Load face recognition data
path = '/home/kenn/face_recog/test'
images = []
classNames = []

myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Variables to track timing for sending messages
lastSentTime = {}
suspiciousSent = False

# Function to encode known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Async functions to send messages to Telegram
async def sendDataToTele(name, img, faceLoc):
    message = f"Face recognized: {name} at {datetime.now()}"
    y1, x1, y2, x2 = faceLoc
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    face_img = img[y1:y2, x1:x2]
    is_success, buffer = cv2.imencode(".jpg", face_img)
    if is_success:
        await application.bot.send_photo(chat_id=CHAT_ID, photo=buffer.tobytes(), caption=message)

async def sendSuspiciousAlert(img, faceLoc):
    message = f"Suspicious face detected at {datetime.now()}"
    y1, x1, y2, x2 = faceLoc
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    face_img = img[y1:y2, x1:x2]
    is_success, buffer = cv2.imencode(".jpg", face_img)
    if is_success:
        await application.bot.send_photo(chat_id=CHAT_ID, photo=buffer.tobytes(), caption=message)

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Load YOLO model for object detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('project/model/best.pt').to(device)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              # ... Add other classes
              ]

cap = cv2.VideoCapture('/home/kenn/face_recog/mencurigakan.mp4')

async def process_video():
    global suspiciousSent
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from video capture")
            break

        # Run YOLO model to detect objects
        result = model(img, stream=True)
        person_detected = False

        for r in result:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass == "person":
                    person_detected = True
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    label = f'{currentClass} {conf:.2f}'
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Proceed with face recognition only if a person is detected
        if person_detected:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            # If no faces are detected, skip face recognition
            if len(facesCurFrame) == 0:
                print("No face detected.")
                continue

            for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                currentTime = datetime.now()

                if matches[matchIndex] and faceDis[matchIndex] < 0.7:
                    name = classNames[matchIndex].upper()
                    y1, x1, y2, x2 = faceLoc
                    y1, x1, y2, x2 = y1 * 4, x1 * 4, y2 * 4, x2 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    if name not in lastSentTime or (currentTime - lastSentTime[name]) > timedelta(minutes=1):
                        lastSentTime[name] = currentTime
                        await sendDataToTele(name, img, (y1, x1, y2, x2))

                    suspiciousSent = False
                elif faceDis[matchIndex] > 0.7 and not suspiciousSent:
                    y1, x1, y2, x2 = faceLoc
                    y1, x1, y2, x2 = y1 * 4, x1 * 4, y2 * 4, x2 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, "SUSPICIOUS", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    await sendSuspiciousAlert(img, (y1, x1, y2, x2))
                    suspiciousSent = True

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

async def main():
    await process_video()

# Start the event loop
asyncio.run(main())

cap.release()
cv2.destroyAllWindows()
