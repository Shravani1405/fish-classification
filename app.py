
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ---------------------------
# 1. Page configuration
# ---------------------------
st.set_page_config(page_title="Fish Classification App", page_icon="🐟", layout="centered")

# ---------------------------
# 2. Load Model
# ---------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fish_model.h5")
    return model

model = load_model()

# ---------------------------
# 3. Class Labels
# ---------------------------
# Replace these with your actual fish class names in the order they were used for training
CLASS_NAMES = [
    "Black Sea Sprat",
    "Gilt-Head Bream",
    "Hourse Mackerel",
    "Red Sea Bream",
    "Sea Bass",
    "Shrimp",
    "Striped Red Mullet",
    "Trout"
]

# ---------------------------
# 4. Helper: Preprocess Image
# ---------------------------
def preprocess_image(img: Image.Image):
    # Resize to match model input size (change if your model uses a different size)
    img = img.resize((224, 224))
    img_array = np.array(img)

    # If grayscale, convert to RGB
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Expand dims for model input
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ---------------------------
# 5. App UI
# ---------------------------
st.title("🐟 Fish Classification App")
st.write("Upload an image of a fish, and the model will predict its species.")

uploaded_file = st.file_uploader("Upload a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = preprocess_image(image)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Show result
    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
