import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Model file name
MODEL_FILE = "pneumonia_cnn_model.h5"

# Google Drive File ID
FILE_ID = "1Q94iWGnw6MXuuYInBZh-Di5x1z-i4CTF"

# Download model if it does not exist
if not os.path.exists(MODEL_FILE):
    st.write("Downloading model...")
    gdown.download(id=FILE_ID, output=MODEL_FILE, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(MODEL_FILE)

# App title
st.title("🩺 Pneumonia Detection from Chest X-Ray")

st.write("Upload a chest X-ray image and the AI model will predict whether it shows Pneumonia or Normal.")

# Upload image
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    # Convert image to RGB
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize
    img = image.resize((150,150))

    # Convert to array
    img_array = np.array(img) / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction button
    if st.button("Predict"):

        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            st.error(f"⚠️ Pneumonia Detected\n\nConfidence: {prediction:.2f}")
        else:
            st.success(f"✅ Normal\n\nConfidence: {1-prediction:.2f}")
