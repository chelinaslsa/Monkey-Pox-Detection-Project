import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from huggingface_hub import hf_hub_download

# Load model
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model_and_metadata():
    metadata_path = 'model_metadata.pkl' 

    model_path = hf_hub_download(
        repo_id="chelinaslsa/monkeypox-resnet50", 
        filename="monkeypox_model_final.keras"
    )
    
    model = tf.keras.models.load_model(model_path, compile=False)    

    if not os.path.exists(metadata_path):
        st.error(f"Metadata file not found: {metadata_path}")
        return None, None
    
    # Load metadata
    metadata = joblib.load(metadata_path)
    return model, metadata

# Load model dan metadata
model, metadata = load_model_and_metadata()

if model is None or metadata is None:
    st.stop()

class_names = metadata['class_names']

# Preprocessing
def preprocess_image(uploaded_image):
    img = uploaded_image.resize(IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# User interface
st.set_page_config(page_title="Mpox Detector", page_icon="🐒")

st.markdown("""
    <h1 style='text-align: center;'>
         Monkeypox Detection 🐒
    </h1>
""", unsafe_allow_html=True)
st.markdown("""
Monkeypox Skin Detection System
This system use a **ResNet50-based Deep Learning model** to classify skin lesions into four categories:
- Acne
- Chickenpox
- Measles
- Monkeypox
""")

uploaded_file = st.file_uploader("Upload skin image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan Gambar
    uploaded_image = Image.open(uploaded_file).convert('RGB')
    
    st.image(uploaded_image, caption='Uploaded Image')
        
    if st.button("Predict"):
        with st.spinner('Preprocessing image...'):
            # Preprocess gambar - FIXED: pakai uploaded_image bukan image
            processed_image = preprocess_image(uploaded_image)
            
            # Prediksi
            predictions = model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100
            
        # Show result
        if predicted_class == "Monkeypox":
            st.error(f"### Detection result: {predicted_class}")
        elif predicted_class == "Measles" or predicted_class == "Chickenpox":
            st.warning(f"### Detection result: {predicted_class}")
        else:
            st.success(f"### Detection result: {predicted_class}")
                
        st.metric("Model confidence score", f"{confidence:.2f}%")
            
        # Grafik Probabilitas
        st.write("---")
        st.write("Class probability:")
        for i, cls in enumerate(class_names):
                prob_percent = predictions[0][i] * 100
                st.write(f"**{cls}**")
                st.progress(int(prob_percent))
                st.caption(f"{prob_percent:.2f}%")
                st.write("")


