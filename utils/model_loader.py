import os
import gdown
import streamlit as st

MODEL_PATH = "model/alzheimer_oasis_model.keras"
FILE_ID    = "1b1X5ZCsoWamEb6XMVgj5xVFWvtbMfpUH"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    import tensorflow as tf
    os.makedirs("model", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... (~330 MB, one time only)"):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model