from helper import *
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Assuming 'predictor' is a function that takes a grayscale image array and returns the classification result
# from predictor import predictor 

# Page configuration
st.set_page_config(page_title="Lung Cancer Classifier", layout="wide")

# Title and Introduction
st.title('Lung Cancer Classifier')
st.markdown("""
    ### Detect Anomalies in Your Lungs
    Utilize our advanced Deep Learning model based on Convolutional Neural Networks (CNN) to analyze X-ray images. 
    The model, trained with a comprehensive dataset from Kaggle, boasts an impressive accuracy of around 90%. 
    Please note: This tool is intended for educational purposes and should not be considered a substitute for professional medical advice.
</s>""")

# Sidebar for user inputs
st.sidebar.header("Upload X-ray Image")
uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])

# Main Content Area
col1, col2 = st.columns([2, 3])  # Adjust the ratio of the columns
with col1:
    if uploaded_file is not None:
        # Convert the file to an Image object and display
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    if uploaded_file is not None:
        # Convert the image to a numpy array and classify
        image_np = np.array(image)
        result = predictor(image_np)
        
        # Use markdown to ensure the text wraps correctly and doesn't get cut off
        st.markdown(f"#### Classification Result")
        st.markdown(f"##### Your chances of having PNEUMONIA is: **{result}**")

# Footer
st.markdown("---")
st.markdown("By Siddartha Koppaka.")
