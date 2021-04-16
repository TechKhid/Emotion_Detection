import os
import cv2 as cv
up_dir = os.getcwd() + r'\facial_expressions-master'
image_dir = os.getcwd() + r'\facial_expressions-master\images'
_dir = os.getcwd() + r'\facial_expressions-master\dataroot'
new_dir = os.mkdir(_dir)
labels = ['sad', 'happy', 'surprise', 'neutral', 'anger']
#label_txt = ['sad.txt', 'happy.txt','surprise.txt', 'neutral.txt', 'anger.txt']
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


def extract(pathway):
    img = cv.imread(pathway)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
    global roi  
    for x, y, w, h in faces:
        color = (0, 45, 0)
        stroke = 2
        cv.rectangle(img_gray, (y, y+h), (x, x+w), color, stroke)
        roi = img_gray[y:y+h, x:x+w]    
    return roi

for label in labels:
    new_folder = os.path.join(_dir, str(label))    
    os.mkdir(new_folder)
    txt_path = os.path.join(up_dir, label +'.txt')
    with open(txt_path, 'r') as f:
        img = [line.strip() for line in f]
    for image in img:
        img_dir = os.path.join(image_dir, image)
        prep_img = extract(img_dir)
        new = os.path.join(new_folder, image)
        cv.imwrite(new, prep_img)
print('Done!')