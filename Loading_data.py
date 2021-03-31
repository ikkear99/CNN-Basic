import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle

datapath = 'dataset/training_set'
CARETORIES = ["cats", "dogs"]
Img_size = 50

training_data = []
def create_training_data():
    for caretory in CARETORIES:
        path = os.path.join(datapath, caretory)
        class_num = CARETORIES.index(caretory)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_arr = cv2.resize(img_arr, (Img_size, Img_size))
                training_data.append([new_arr,class_num])
            except Exception as e:
                pass

create_training_data()
print(len(training_data))

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
print(X[1])



