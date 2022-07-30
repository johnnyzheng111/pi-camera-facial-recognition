import os

TRAINER_DIR = os.path.dirname(os.path.abspath(__file__)) #gets the dir of whereever trainer.py is

faces_dir = os.path.join(TRAINER_DIR,"faces") #already have the path of trainer file, adds face dir onto it

for root, dirs, files in os.walk(faces_dir):
    for file in files:
        if file.endswith('jpg'):
            path = os.path.join(root, file)
            name = os.path.basename(root)
            print(name, path)