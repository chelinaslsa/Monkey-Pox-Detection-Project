import streamlit as st
st.set_page_config(page_title="Mpox Detector", page_icon="🐒")

import numpy as np
import joblib
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from huggingface_hub import hf_hub_download

IMG_SIZE = (224, 224)

@st.cache_resource
def load_model_and_metadata():
    metadata_path = "model_metadata.pkl"

    model_path = hf_hub_download(
        repo_id="celinasalsa/monkeypox-resnet50",
        filename="monkeypox_model_final.keras"
    )

    model = tf.keras.models.load_model(model_path, compile=False)

    if not os.path.exists(metadata_path):
        st.stop()

    metadata = joblib.load(metadata_path)
    return model, metadata

model, metadata

