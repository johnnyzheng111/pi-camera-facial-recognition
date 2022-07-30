import cv2
from datetime import datetime, timedelta
# from gpiozero import MotionSensor
from time import sleep
import numpy as np
import pickle

#TODO
# video -> recognize face in video -> train facial recognition
# Video: Use OpenCV video capture 
# 
# Recognize face in video: Using Haar Cascade Classifiers, we are able to detect motion of specific
# parts of the face. We are currently detecting frontalface 
#
#
#
face_casade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')
eye_Cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

names = {}
with open("names.pickle", 'rb') as f:
    names_ = pickle.load(f)
    names = {v:k for k,v in names_.items()} #inverts dictionary 

cap = cv2.VideoCapture(0) #capturing video // arg 0 -> takes in your default webcam

#set the capture rate to 480p to not overload the raspi
def make_480p():
    cap.set(3, 640) #arg 3 - > width
    cap.set(4, 480) #arg 4 - > height
def detect_face(frame,scale,min):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts bgr to gray
    #Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
    faces = face_casade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=min) 

    for x,y,w,h in faces:
        #(x,y,w,h) all of the values of where the face is (region of interest)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45: #and conf <=85:
            print(id_)
            print(names[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = names[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA) 
        c = (0,255,0)
        stroke = 2
        width = x+w 
        height = y+h
        cv2.rectangle(frame,(x,y), (width,height),c,stroke)
        eyes = eye_Cascade.detectMultiScale(roi_gray)
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0),2)
            print("hello")



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