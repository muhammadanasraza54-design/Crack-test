import streamlit as st
import numpy as np
from PIL import Image
import os

import tensorflow as tf

model_path = 'TCF_Final_Crack_Modeel_11_April_2026.h5'

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)

else:
    st.error(f"Model file '{model_path}' nahi mili!")

# Image upload function
file = st.file_uploader("Please upload an image", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (224, 224) # Jo size aapne training mein use kiya tha (e.g., 224x224)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img_array = np.asarray(image)
    img_reshape = img_array[np.newaxis, ...]
    
    # Preprocessing (agar training mein scaling ki thi)
    img_reshape = img_reshape / 255.0
    
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    # Prediction button
    if st.button("Detect Crack"):
        with st.spinner('Analyzing...'):
            predictions = import_and_predict(image, model)
            
            # Agar binary classification hai (Crack vs No Crack)
            if predictions[0][0] > 0.5:
                st.error(f"⚠️ Crack Detected! (Confidence: {predictions[0][0]:.2%})")
            else:
                st.success(f"✅ No Crack Detected. (Confidence: {1 - predictions[0][0]:.2%})")
