import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tflite_runtime.interpreter as tflite
import os

st.title("🏗️ TCF Building Crack Detection")

# TFLite model ka sahi naam jo aapne push kiya hai
model_path = 'model.tflite'

def predict(image_data):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Image Preprocessing
    size = (224, 224)
    img = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img).astype('float32') / 255.0
    img_reshape = img_array[np.newaxis, ...]

    interpreter.set_tensor(input_details[0]['index'], img_reshape)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

file = st.file_uploader("School ki photo upload karein", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, use_container_width=True)
    if st.button("🔍 Analyze Structure"):
        if os.path.exists(model_path):
            score = predict(image)
            if score > 0.5:
                st.error(f"⚠️ Crack Detected! (Confidence: {score:.2%})")
            else:
                st.success(f"✅ Structure Safe (Confidence: {1-score:.2%})")
        else:
            st.error("Model file nahi mili!")
