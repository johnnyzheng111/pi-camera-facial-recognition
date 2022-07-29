import cv2
import face_recognition

face_casade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cam = cv2.VideoCapture(0)

while True:
    ret , frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_casade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for x,y,w,h in faces:
        color = (0,255,0)
        
        stroke =2
        width = x+w
        height = y+h
        cv2.rectangle(frame,(x,y), (width,height),color,stroke)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):break

cam.release()
cv2.destroyAllWindows()