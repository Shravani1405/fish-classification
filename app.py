# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ==============================
# 1. Page Configuration
# ==============================
st.set_page_config(
    page_title="Multiclass Fish Image Classification",
    page_icon="🐟",
    layout="centered"
)

# ==============================
# 2. Load Model
# ==============================
@st.cache_resource
def load_model():
    model_path = "best_fish_model.h5"  # Ensure this file is in same folder as app.py
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# ==============================
# 3. Class Labels
# ==============================
# Replace with your dataset's actual class names in correct order
CLASS_NAMES = [
    "class_1", "class_2", "class_3", "class_4", "class_5"
    # Example: "Salmon", "Tuna", "Catfish", ...
]

# ==============================
# 4. Helper Functions
# ==============================
def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))  # same as training size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img: Image.Image):
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence_scores = predictions[0]
    return predicted_index, confidence_scores

# ==============================
# 5. UI - Title & Info
# ==============================
st.title("🐟 Multiclass Fish Image Classification")
st.markdown("""
### Project Overview
This application classifies fish images into multiple categories using a deep learning model (EfficientNetB4).  
The model was trained with transfer learning and fine-tuning on a custom fish dataset.
""")

st.info("Upload an image of a fish and get its predicted category with confidence scores.")

# ==============================
# 6. File Uploader
# ==============================
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prediction
    with st.spinner("Predicting..."):
        pred_index, confidence_scores = predict_image(image)

    st.success(f"**Predicted Class:** {CLASS_NAMES[pred_index]}")
    
    # Show confidence scores
    st.subheader("Confidence Scores")
    for i, score in enumerate(confidence_scores):
        st.write(f"{CLASS_NAMES[i]}: {score*100:.2f}%")

# ==============================
# 7. Footer
# ==============================
st.markdown("""
---
**Skills Used:** Deep Learning, Python, TensorFlow/Keras, Transfer Learning, Data Augmentation, Model Deployment with Streamlit.  
**Author:** Your Name  
**Domain:** Image Classification  
""")

