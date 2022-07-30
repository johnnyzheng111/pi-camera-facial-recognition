import cv2
from datetime import datetime, timedelta
# from gpiozero import MotionSensor
from time import sleep
import numpy as np

#TODO
# video -> recognize face in video -> train facial recognition
# Video: Use OpenCV video capture 
# 
# Recognize face in video: Using Haar Cascade Classifiers, we are able to detect motion of specific
# parts of the face. We are currently using frontalface 
#
#
#
face_casade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0) #capturing video // arg 0 -> takes in your default webcam

#set the capture rate to 480p to not overload the raspi
def make_480p():
    cap.set(3, 640) #arg 3 - > width
    cap.set(4, 480) #arg 4 - > height
def detect_face(frame,scale,min):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts rgb to gray
    #Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
    faces = face_casade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=min) 
    for x,y,w,h in faces:
        #(x,y,w,h) all of the values of where the face is (region of interest)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        c = (0,255,0)
        stroke = 2
        width = x+w 
        height = y+h
        cv2.rectangle(frame,(x,y), (width,height),c,stroke)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    detect_face(frame,1.5,5)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()