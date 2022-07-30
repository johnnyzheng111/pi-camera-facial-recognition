import os
import cv2
from PIL import Image

cap = cv2.VideoCapture(0)

name = input("Enter first name: ")
name.lower()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
name_path = os.path.join(CURRENT_DIR,f"src/faces/{name}")
list_ = None
num_of_files = None
try:
    list_ = os.listdir(name_path)
    num_of_files = len(list_)+1
except:
    print("Cannot find path")
    exit()

while True:
    ret, frame = cap.read()
    cv2.imshow('ESC to exit, Space to take picture',frame)
    k = cv2.waitKey(0)
    if k==27: #esc
        break
    elif k == 32: #space
        cv2.imwrite(os.path.join(name_path,f'{num_of_files}.jpg'),frame)
        # img = Image.open(os.path.join(name_path,f'{num_of_files}.jpg')).convert('L')
        # img.save(os.path.join(name_path,f'{num_of_files}.jpg'))
        num_of_files+=1
cap.release()
cv2.destroyAllWindows()
