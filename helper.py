import requests
import preprocessdata
from keras.models import load_model
import pandas as pd
import numpy as np

cells = ['PNEUMONIA','NORMAL']

loaded_model = load_model('Static/lcd.pkl')


def ValuePredictor(img):
    image = preprocessdata.preprocess(img)
    # print(image)
    result = loaded_model.predict(image)
    return result

def predictor(img):
    prediction = ValuePredictor(img)

    pred_f = prediction[0]

    pred_fr = pred_f[0]

    pred = 1 - float(pred_fr)
    pred = round(pred*100 , 2)

    if float(pred_fr) < 0.7 :
        if float(pred_fr) < 0.25 :
            result = f'Your chances of having PNUEMONIA is around {pred} %. Your chances are too high, Visit a Doctor Immediately!'
        else :
            result = f'Your chances of having PNUEMONIA is around {pred} %. You may have PNUEMONIA cells, Visit a Doctor!'
    if float(pred_fr) >= 0.7 :
        result = f'Your chances of having PNUEMONIA is around {pred} %. You are fine, maintain good health to avoid problems in future.'
    
    return result

# print(predictor('sample.jpeg'))

    