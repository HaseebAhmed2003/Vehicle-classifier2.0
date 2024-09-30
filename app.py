import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
  
def predict(uploaded_file, model, classes):
    img = Image.open(uploaded_file)
    img = img.resize((300, 300))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = classes[predicted_class_index]

    st.write(f"Predicted Vehicle: {predicted_class_label}")
    st.image(img, use_column_width=True)

def run():
    st.header('Vehicle Type Recognition 	:busstop:')
    st.write('The objective of this project is to build a machine learning model to classify vehicles into the following categories using Convolutional Neural Networks.')
    st.markdown("""
                - Auto Rickshaw  :auto_rickshaw:
                - Bicycle  :bicyclist:
                - Bus  :bus:
                - Car  :car:
                - Motorcycle  :racing_motorcycle:
                - Truck  :truck:
                - Van 	:minibus:
                """)

    with st.form(key='Form Upload Vehicle Type Recognition'):
        uploaded_files = st.file_uploader("Choose a .JPEG/.JPG/.PNG file", accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:

                st.write("filename:", uploaded_file.name)
                model = load_model('vehicle_recognition_model.keras')
                classes = ['Auto-rickshaw :auto_rickshaw:', 'Bicycle :bicyclist:', 'Bus :bus:', 'Car :car:', 'Motorcycle :racing_motorcycle:', 'Truck :truck:', 'Van 	:minibus:']
                predict(uploaded_file, model, classes)

        st.form_submit_button(label='Submit')

if __name__ == '__main__':
    run()