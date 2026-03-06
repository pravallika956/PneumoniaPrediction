import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

MODEL_FILE = "pneumonia_cnn_model.h5"

# Google Drive File ID
FILE_ID = "1Q94iWGnw6MXuuYInBZh-Di5x1z-i4CTF"

# Download model if not present
if not os.path.exists(MODEL_FILE):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_FILE)

st.title("🩺 Pneumonia Detection from Chest X-Ray")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((150,150))
    img = np.array(img)/255.0
    img = img.reshape(1,150,150,3)

    if st.button("Predict"):
        prediction = model.predict(img)

        if prediction[0][0] > 0.5:
            st.error("⚠️ Pneumonia Detected")
        else:
            st.success("✅ Normal")