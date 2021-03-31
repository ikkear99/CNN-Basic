import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

datapath = 'dataset/test_set'
CARETORIES = ["cats", "dogs"]

for caretory in CARETORIES:
    path = os.path.join(datapath, caretory)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_arr, cmap='gray')
        plt.show()
        new_arr = cv2.resize(img_arr,(50,50))
        plt.imshow(new_arr, cmap = 'gray')
        plt.show()

training_data = []
def create_traning_data():
    pass
