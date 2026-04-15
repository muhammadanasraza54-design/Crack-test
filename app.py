import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os

# TFLite Import with Fallback
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        st.error("❌ Critical Error: TFLite Runtime or TensorFlow not found. Please check requirements.txt")

# Page Configuration
st.set_page_config(page_title="TCF Crack Detection", page_icon="🏗️")
st.title("🏗️ TCF Building Crack Detection")
st.write("School buildings ki structural safety check karne ke liye AI portal.")

model_path = 'model.tflite'

# Model Loader with Cache (App ki speed barhane ke liye)
@st.cache_resource
def get_interpreter():
    if os.path.exists(model_path):
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

def predict(image_data, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Image Preprocessing
    size = (224, 224)
    img = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img).astype('float32') / 255.0
    img_reshape = img_array[np.newaxis, ...]

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_reshape)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# UI Setup
file = st.file_uploader("🏫 School ki photo upload karein", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded View", use_container_width=True)
    
    if st.button("🔍 Analyze Structure"):
        interpreter = get_interpreter()
        
        if interpreter is not None:
            with st.spinner('AI analysis chal raha hai...'):
                try:
                    score = predict(image, interpreter)
                    
                    # Result Display Logic
                    if score > 0.5:
                        st.error(f"⚠️ **Crack Detected!** (Confidence: {score:.2%})")
                        st.warning("Recommendation: Maintenance team ko notify karein.")
                    else:
                        st.success(f"✅ **Structure Safe.** (Confidence: {1-score:.2%})")
                except Exception as e:
                    st.error(f"Prediction mein masla aaya: {e}")
        else:
            st.error(f"❌ Error: Model file '{model_path}' repository mein nahi mili.")
