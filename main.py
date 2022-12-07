from helper import *

#importing all the helper fxn from helper.py which we will create later
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_theme(style="darkgrid")

sns.set()

st.title('Lung Cancer Classifier')

def save_uploaded_file(uploaded_file):

    try:

        with open(os.path.join('static/images',uploaded_file.name),'wb') as f:

            f.write(uploaded_file.getbuffer())

        return 1    

    except:

        return 0


components.html("""<html  style="font-family:'Poppins', sans-serif;">
    <h1> Detect Anamolies in your Lungs using this Lung Cancer Detector </h1>
    
    <p style="color: grey;">This Deep Learning model is based on CNN. We used a dataset imported from Kaggle, and prepared
    a CNN model with an accuracy around 90%. Later, using Streamlit the model is deployed to web.<br><br>
    <small>This project is made for Educational purposes, not reliable for real world issues.</small>
    </p>
    </html>""" ,height= 400)



uploaded_file = st.file_uploader("Upload Image")




# text over upload button "Upload Image"

if uploaded_file is not None:

    if save_uploaded_file(uploaded_file): 

        # display the image

        display_image = Image.open(uploaded_file)

        st.image(display_image)

        result = predictor(os.path.join('static/images',uploaded_file.name)) 
        
        os.remove('static/images/'+uploaded_file.name)

        st.write(result)