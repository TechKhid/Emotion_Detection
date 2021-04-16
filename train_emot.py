import cv2 as cv
import numpy as np
import pickle
import os

base_dir = base_dir = os.path.dirname(os.path.abspath(__file__))  
_dir = os.path.join(base_dir, "facial_expressions-master")
face_dir = os.path.join(_dir, "dataroot")
labels = ['sad', 'happy', 'surprise', 'neutral', 'anger']

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()


def resize(path, scale):
    width = int(path.shape[1] * scale)
    height = int(path.shape[0] * scale)
    dimension = (width, height)
    return cv.resize(path, dimension)


x_train = []
y_labels = []
label_ids = {}
current_id = 0



for root, dirs, files in os.walk(face_dir):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-")
            print(label, path)
            fetch = cv.imread(path)
            resize_img = resize(fetch, 0.70)
            gray = cv.cvtColor(resize_img, cv.COLOR_BGR2GRAY)
            img_array = np.array(gray, 'uint8')
            #print(img_array)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1                  
            id_ = label_ids[label]
            #print(label_ids)
            faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.2, minNeighbors=4)
            for x, y, w, h in faces:
                roi = img_array[y:y+h, x:x+w]
                x_train.append(roi)                
                y_labels.append(id_)
                print(x_train, y_labels)


with open('Emolabel.pickles', 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('emo_trainer.yml')


            







