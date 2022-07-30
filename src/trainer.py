import os
import numpy as np
from PIL import Image
import cv2
import pickle

TRAINER_DIR = os.path.dirname(os.path.abspath(__file__)) #gets the dir of whereever trainer.py is
faces_dir = os.path.join(TRAINER_DIR,"faces") #already have the path of trainer file, adds face dir onto it

face_casade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')
eye_Cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_id = {}
names = []
paths = []

for root, dirs, files in os.walk(faces_dir):
    for file in files:
        if file.endswith('jpg'):
            path = os.path.join(root, file)
            name = os.path.basename(root)
            # paths.append(path)
            # names.append(name)
            if not name in label_id:
                label_id[name] = current_id
                current_id +=1
            id_ = label_id[name]
            #takes every pixel and turns it into numpy array // convert image to numbers
            pil_img = Image.open(path).convert("L") # converts gray
            size = (550, 550)
            final_image = pil_img.resize(size, Image.Resampling.LANCZOS)
            img_arr = np.array(final_image, "uint8")
            faces = face_casade.detectMultiScale(img_arr,scaleFactor=1.5, minNeighbors=5)
            eyes = eye_Cascade.detectMultiScale(img_arr,scaleFactor=1.5, minNeighbors=5)

            for x,y,w,h in faces:
                roi = img_arr[y:y+h, x:x+w]
                names.append(id_)
                paths.append(roi)
                eyes = eye_Cascade.detectMultiScale(roi)
                for (ex,ey,ew,eh) in eyes:
                    names.append(id_)
                    paths.append(roi)
            # for (ex,ey,ew,eh) in eyes:
            #     roi = img_arr[ey:ey+eh, ex:ex+ew]
            #     names.append(id_)
            #     paths.append(roi)
with open("names.pickle",'wb')as f:
    pickle.dump(label_id,f)
recognizer.train(paths,np.array(names))
recognizer.save("trainer.yml")