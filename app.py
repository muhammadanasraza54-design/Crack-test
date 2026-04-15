import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os
import tflite_runtime.interpreter as tflite

st.set_page_config(page_title="TCF Crack Detection")
st.title("🏗️ Building Crack Detection Portal")

# TFLite Model ka naam (Ye file aapko abhi banani hogi niche step mein)
model_path = 'model.tflite'

def predict_tflite(image_data):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocessing
    size = (224, 224)
    img = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img).astype('float32') / 255.0
    img_reshape = img_array[np.newaxis, ...]
    
    interpreter.set_tensor(input_details[0]['index'], img_reshape)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction[0][0]

file = st.file_uploader("Upload School Image", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, use_container_width=True)
    
    if st.button("Analyze"):
        if os.path.exists(model_path):
            score = predict_tflite(image)
            if score > 0.5:
                st.error(f"⚠️ Crack Detected! ({score:.2%})")
            else:
                st.success(f"✅ Safe: No Crack ({1-score:.2%})")
        else:
            st.warning("Pehle model.tflite file upload karein.")
