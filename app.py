import streamlit as st
import numpy as np
from PIL import Image, ImageOps # ImageOps zaroori hai prediction ke liye
import os
import tensorflow as tf

# TCF Branding & UI
st.set_page_config(page_title="TCF Crack Detection", page_icon="🏗️")
st.title("🏗️ Building Crack Detection Portal")
st.write("TCF school buildings ki safety ke liye automatic crack detection system.")

# Model path jo aapke GitHub par hai
model_path = 'TCF_Final_Crack_Modeel_11_April_2026.h5'

# Model loading logic with Cache taake app fast chale
@st.cache_resource
def load_my_model():
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_my_model()

if model is None:
    st.error(f"❌ Model file '{model_path}' nahi mili!")
else:
    # Image upload UI
    file = st.file_uploader("School ki image upload karein", type=["jpg", "png", "jpeg"])

    def import_and_predict(image_data, model):
        size = (224, 224) 
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image)
        img_reshape = img_array[np.newaxis, ...]
        
        # Preprocessing (Scaling)
        img_reshape = img_reshape.astype('float32') / 255.0
        
        prediction = model.predict(img_reshape)
        return prediction

    if file is not None:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Analyze Structure"):
            with st.spinner('Processing...'):
                predictions = import_and_predict(image, model)
                score = predictions[0][0]
                
                # Result display
                if score > 0.5:
                    st.error(f"⚠️ **Crack Detected!** (Confidence: {score:.2%})")
                    st.warning("Zaroori: Engineering department ko notify karein.")
                else:
                    st.success(f"✅ **No Major Crack Detected.** (Confidence: {1 - score:.2%})")
