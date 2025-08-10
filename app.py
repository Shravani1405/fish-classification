import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="Fish Classification App", layout="centered")

# Load model once using Streamlit cache
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fish_model.h5")
    return model

model = load_model()

# Class names - replace with your actual fish species
CLASS_NAMES = ['Salmon', 'Tuna', 'Trout', 'Mackerel', 'Catfish']

# App title
st.title("🐟 Fish Classification using Deep Learning")
st.write("Upload an image of a fish and let the AI classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (224, 224))  # Adjust size as per your model
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_index = np.argmax(score)
    confidence = 100 * np.max(score)

    st.subheader(f"Prediction: **{CLASS_NAMES[class_index]}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
