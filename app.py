import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os

# 1. TFLite Safe Import
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        st.error("❌ Critical Error: TFLite Runtime or TensorFlow not found.")

# 2. Page Configuration
st.set_page_config(page_title="TCF Crack Detection", page_icon="🏗️")
st.title("🏗️ TCF Building Crack Detection")
st.write("School buildings ki structural safety check karne ke liye AI portal.")

# Model file ka naam
model_path = 'model.tflite'

# 3. Model Loader with Cache
@st.cache_resource
def get_interpreter():
    if os.path.exists(model_path):
        try:
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            st.error(f"Model load karne mein masla: {e}")
            return None
    return None

# 4. Prediction Function
def predict(image_data, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Model requirement: 120x120 pixels
    size = (120, 120) 
    
    img = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img).astype('float32') / 255.0
    img_reshape = img_array[np.newaxis, ...]

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_reshape)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# 5. UI Setup
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
                    
                    # 6. Result Display Logic
                    if score > 0.5:
                        st.error(f"⚠️ **Crack Detected!**")
                        
                        # Visual clarity columns
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("AI Confidence", f"{score:.2%}")
                        with col2:
                            # Severity based on confidence
                            severity = "High" if score > 0.8 else "Medium"
                            st.metric("Severity Level", severity)

                        st.subheader("📊 Analysis Summary:")
                        st.write(f"* **Detection Probability:** 100 mein se {score*100:.1f} hissa imkan hai ke ye structural defect hai.")
                        st.write(f"* **Urgency:** {'Fauri munaayna (inspection) zaroori hai.' if score > 0.8 else 'Routine maintenance mein check karein.'}")
                        
                        st.warning("📋 **Recommendation:** Maintenance team ko notify karein aur structural stability report check karein.")
                    else:
                        st.success(f"✅ **Structure Safe.**")
                        st.info(f"AI Prediction: 100 mein se {(1-score)*100:.1f}% imkan hai ke structure mehfooz hai.")
                
                except Exception as e:
                    st.error(f"Prediction mein masla aaya: {e}")
        else:
            st.error(f"❌ Error: Model file '{model_path}' nahi mili ya load nahi ho saki.")
