import cv2
import numpy as np
import face_recognition

imgInput = face_recognition.load_image_file('syiar.jpg')
imgInput = cv2.cvtColor(imgInput,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('dika.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgInput)[0]
encodeInput = face_recognition.face_encodings(imgInput)[0]
cv2.rectangle(imgInput,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,255,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,255,255),2)

result = face_recognition.compare_faces([encodeInput],encodeTest)
faceDis = face_recognition.face_distance([encodeInput],encodeTest)
print(result,faceDis)

cv2.putText(imgTest,f'{result} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

cv2.imshow('imgInput',imgInput)
cv2.imshow('imgTest',imgTest)
cv2.waitKey(0)