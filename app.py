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
                    
                    # 6. Behtar Result Display Logic
                    if score > 0.5:
                        st.error(f"⚠️ **Crack Detected!**")
                        
                        # Calculation for visual clarity
                        crack_percentage = score * 100
                        # Pixels se size ka andaza (approximate scale)
                        estimated_width = "Major" if score > 0.85 else "Minor"
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("AI Confidence", f"{score:.2%}")
                        with col2:
                            st.metric("Crack Intensity", f"{crack_percentage:.1f}%")
                        with col3:
                            st.metric("Crack Type", estimated_width)

                        st.subheader("📊 Detailed Analysis Summary:")
                        
                        # Parhne walon ke liye asaan points
                        st.write(f"1. **Severity:** Deewar par crack ki shiddat **{crack_percentage:.1f}%** record ki gayi hai.")
                        st.write(f"2. **Physical Observation:** Ye aik **{estimated_width}** structural defect maloom hota hai.")
                        st.write(f"3. **Risk Level:** {'Ziyada (High)' if score > 0.8 else 'Darmayana (Medium)'}.")
                        st.write(f"4. **Urgency:** {'Fauri munaayna (inspection) aur repair zaroori hai.' if score > 0.8 else 'Routine maintenance checklist mein shamil karein.'}")
                        
                        st.divider()
                        st.warning("📋 **Engineer's Recommendation:** Maintenance team ko is school ka portal reference de kar notify karein aur structural stability report check karein.")
                    else:
                        st.success(f"✅ **Structure Safe.**")
                        st.info(f"AI Analysis: 100 mein se {(1-score)*100:.1f}% imkan hai ke structure mein koi wazeh crack nahi hai.")
                
                except Exception as e:
                    st.error(f"Prediction mein masla aaya: {e}")
        else:
            st.error(f"❌ Error: Model file '{model_path}' nahi mili ya load nahi ho saki.")
