import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Google Drive settings
MODEL_PATH = "trained_model.keras"
FILE_ID = "14wfSOKjEMYo92PG8D1tatbStX_GZCupx"

def download_model(file_id, output_path):
    # this URL format should force a direct download
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    # for large file, add confirm parameter
    url_with_confirm = f"{url}&confirm=t"
    gdown.download(url_with_confirm, output_path, quiet=False)

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait ⏳"):
        download_model(FILE_ID, MODEL_PATH)

# Debug: show file size
if os.path.exists(MODEL_PATH):
    st.write("Downloaded file size (bytes):", os.path.getsize(MODEL_PATH))
else:
    st.error("Model file does not exist after download.")

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# your class names etc
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    # ... rest of your classes ...
    'Tomato___healthy'
]

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to predict the plant disease.")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.success(f"✅ Predicted: **{predicted_class}** with {confidence:.2f}% confidence")
