import os
import gdown
import streamlit as st

MODEL_PATH = "model/alzheimer_model.tflite"
FILE_ID    = "13xmTqPimM5lGqDLr_uq3Z-F4A2S_t9uP"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    os.makedirs("model", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... (one-time only)"):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter
