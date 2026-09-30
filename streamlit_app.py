import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Google Drive settings
MODEL_PATH = "trained_model.keras"
FILE_ID = "14wfSOKjEMYo92PG8D1tatbStX_GZCupx"

# Must match the notebook: image_dataset_from_directory(image_size=(128, 128))
# with raw 0-255 pixels (the model has no Rescaling layer).
IMAGE_SIZE = (128, 128)

def download_model(file_id, output_path):
    # this URL format should force a direct download
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    # for large file, add confirm parameter
    url_with_confirm = f"{url}&confirm=t"
    gdown.download(url_with_confirm, output_path, quiet=False)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... Please wait ⏳"):
            download_model(FILE_ID, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

# Load model once and reuse it across reruns
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Same order as validation_set.class_names in plantdetection.ipynb (38 classes)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot',
    'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
]

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to predict the plant disease.")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.success(f"✅ Predicted: **{predicted_class}** with {confidence:.2f}% confidence")
