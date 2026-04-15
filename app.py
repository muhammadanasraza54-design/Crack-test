import streamlit as st
import numpy as np
from PIL import Image, ImageOps  # <--- ImageOps yahan add kar diya hai
import os
import tensorflow as tf

# Page Title & Header
st.title("🏗️ TCF Building Crack Detection")
st.write("School buildings mein cracks detect karne ke liye image upload karein.")

model_path = 'TCF_Final_Crack_Modeel_11_April_2026.h5'

# Model loading logic
@st.cache_resource # Taake model baar baar load na ho aur app slow na ho
def load_my_model():
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        return None

model = load_my_model()

if model is None:
    st.error(f"❌ Model file '{model_path}' nahi mili! Please check karein ke file folder mein maujood hai.")
else:
    # Image upload function
    file = st.file_uploader("Please upload an image", type=["jpg", "png", "jpeg"])

    def import_and_predict(image_data, model):
        size = (224, 224) 
        # Image resize and fit
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image)
        
        # Newaxis add karna
        img_reshape = img_array[np.newaxis, ...]
        
        # Normalization (Aapne /255.0 use kiya hai, check karein training mein yehi tha?)
        img_reshape = img_reshape.astype('float32') / 255.0
        
        prediction = model.predict(img_reshape)
        return prediction

    if file is None:
        st.info("ℹ️ Please upload an image file to start.")
    else:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Prediction button
        if st.button("Detect Crack"):
            with st.spinner('🔍 Analyzing Building Surface...'):
                predictions = import_and_predict(image, model)
                
                # Confidence score nikalna
                score = predictions[0][0]
                
                if score > 0.5:
                    st.error(f"⚠️ **Crack Detected!** (Confidence: {score:.2%})")
                else:
                    st.success(f"✅ **No Crack Detected.** (Confidence: {1 - score:.2%})")
