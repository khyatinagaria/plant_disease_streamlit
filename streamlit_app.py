import streamlit as st
import tensorflow as tf
import gdown
import os

MODEL_PATH = "trained_model.keras"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        url = "https://drive.google.com/file/d/14wfSOKjEMYo92PG8D1tatbStX_GZCupx/view?usp=drive_link"  # replace with your file id
        gdown.download(url, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

st.title('🎈 App Name')

st.write('Hello world!')
