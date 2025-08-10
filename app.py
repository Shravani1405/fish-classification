import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# -----------------------
# 1. App Title
# -----------------------
st.set_page_config(page_title="Fish Classification App", layout="centered")
st.title("Fish Classification with EfficientNetB4")
st.write("Upload a fish image, and the model will classify it into the correct category.")

# -----------------------
# 2. Load Model and Classes
# -----------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_fish_model.h5")  # change to your saved model file
    return model

model = load_model()

# Load class names (ensure you save a class_indices.json after training)
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse the dictionary to map index → class name
class_names = {v: k for k, v in class_indices.items()}

# -----------------------
# 3. Preprocessing Function
# -----------------------
def preprocess_image(image):
    img = image.resize((380, 380))  # EfficientNetB4 input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale
    return img_array

# -----------------------
# 4. File Uploader
# -----------------------
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    st.write("Processing and predicting...")
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]

    # Get top prediction
    top_index = np.argmax(predictions)
    predicted_class = class_names[top_index]
    confidence = predictions[top_index] * 100

    # -----------------------
    # 5. Show Results
    # -----------------------
    st.markdown(f"### Predicted Class: **{predicted_class}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # Confidence bar chart
    st.subheader("Confidence Scores for All Classes")
    st.bar_chart(predictions)
