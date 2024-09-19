import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
from telegram import Bot
from telegram.ext import Application
import asyncio

# Define your Telegram bot token and chat ID
BOT_TOKEN = '7270440999:AAGHWa8CdB3oL5kR3iwMsj0QgfSyi77RiXU'
CHAT_ID = '7513725643'

# Initialize the application
application = Application.builder().token(BOT_TOKEN).build()

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

# Dictionary to keep track of when a face was last recognized
lastSentTime = {}
# Dictionary to track if a suspicious alert has been sent
suspiciousSent = False

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Asynchronous function to send recognized data and image to Telegram
async def sendDataToTele(name, img, faceLoc):
    message = f"Face recognized: {name} at {datetime.now()}"
    y1, x1, y2, x2 = faceLoc

    # Correct face location if needed
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    if x2 > x1 and y2 > y1:
        face_img = img[y1:y2, x1:x2]
        is_success, buffer = cv2.imencode(".jpg", face_img)
        if is_success:
            await application.bot.send_photo(chat_id=CHAT_ID, photo=buffer.tobytes(), caption=message)
        else:
            print("Failed to encode image.")
    else:
        print("Invalid face location, skipping sending image.")

# Asynchronous function to send suspicious face alert to Telegram
async def sendSuspiciousAlert(img, faceLoc):
    message = f"Suspicious face detected at {datetime.now()}"
    y1, x1, y2, x2 = faceLoc

    # Correct face location if needed
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    if x2 > x1 and y2 > y1:
        face_img = img[y1:y2, x1:x2]
        is_success, buffer = cv2.imencode(".jpg", face_img)
        if is_success:
            await application.bot.send_photo(chat_id=CHAT_ID, photo=buffer.tobytes(), caption=message)
        else:
            print("Failed to encode image.")
    else:
        print("Invalid face location, skipping sending image.")

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture('/home/kenn/face_recog/vid4.mp4')

async def process_video():
    global suspiciousSent
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from video capture")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        # If no faces are detected, skip to the next frame
        if len(facesCurFrame) == 0:
            print("No face detected.")
            continue

        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            currentTime = datetime.now()

            if matches[matchIndex] and faceDis[matchIndex] < 0.7:
                # Recognized face
                name = classNames[matchIndex].upper()
                print(name)
                y1, x1, y2, x2 = faceLoc
                y1, x1, y2, x2 = y1*4, x1*4, y2*4, x2*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # Check if message hasn't been sent in the last minute
                if name not in lastSentTime or (currentTime - lastSentTime[name]) > timedelta(minutes=1):
                    lastSentTime[name] = currentTime
                    await sendDataToTele(name, img, (y1, x1, y2, x2))

                # Reset suspicious flag since a recognized face was found
                suspiciousSent = False

            elif faceDis[matchIndex] > 0.7 and not suspiciousSent:
                # Suspicious face, send alert only if it hasn't been sent yet
                print("Suspicious!")
                y1, x1, y2, x2 = faceLoc
                y1, x1, y2, x2 = y1*4, x1*4, y2*4, x2*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "SUSPICIOUS", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # Send alert for suspicious activity and set flag to prevent resending
                await sendSuspiciousAlert(img, (y1, x1, y2, x2))
                suspiciousSent = True

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

async def main():
    # Start video processing
    await process_video()

# Run the main function in a single event loop
asyncio.run(main())

cap.release()
cv2.destroyAllWindows()
