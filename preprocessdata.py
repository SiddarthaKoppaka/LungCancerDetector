import numpy as np
import pickle
import cv2

img_size = 150


def get_training_data(img):
    data = []
    img_arr = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    resized_arr = cv2.resize(img_arr,(img_size,img_size))
    data.append([resized_arr])
    return np.array(data)


def preprocess(img):
    tesst = get_training_data(img)
    ptest = []

    for feature in tesst:
        ptest.append(feature)

    ptest = np.array(ptest) / 255
    ptest = ptest.reshape(-1, img_size, img_size, 1)

    return ptest
