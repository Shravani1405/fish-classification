
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import tempfile
import os

# -------------------
# App Configuration
# -------------------
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")

# -------------------
# Load Model
# -------------------
@st.cache_resource
def load_trained_model():
    model_path = "brain_tumor_model.h5"  # Place your trained model in same folder
    model = load_model(model_path)
    return model

model = load_trained_model()

# -------------------
# Preprocessing Function
# -------------------
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize
    return img_array

# -------------------
# Prediction Function
# -------------------
def predict_image(model, img_array):
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]
    return class_index, confidence, predictions[0]

# -------------------
# Class Labels (update these as per your dataset)
# -------------------
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# -------------------
# Streamlit UI
# -------------------
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image to classify it into one of the tumor categories.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    # Display image
    st.image(temp_file.name, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess & Predict
    with st.spinner("Classifying..."):
        img_array = preprocess_image(temp_file.name)
        class_index, confidence, all_confidences = predict_image(model, img_array)

    # Show Results
    st.success(f"Prediction: **{CLASS_NAMES[class_index]}**")
    st.info(f"Confidence: **{confidence*100:.2f}%**")

    # Show all class probabilities
    st.subheader("Class Probabilities")
    for i, score in enumerate(all_confidences):
        st.write(f"{CLASS_NAMES[i]}: {score*100:.2f}%")

    # Optional: Grad-CAM (if you want explainability)
    if st.checkbox("Show Grad-CAM Heatmap"):
        # Grad-CAM implementation
        img = cv2.imread(temp_file.name)
        img_resized = cv2.resize(img, (224, 224))
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(index=-3).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.array([img_resized / 255.0]))
            loss = predictions[:, class_index]
        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        # Overlay heatmap
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

        st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)

    # Clean up
    os.unlink(temp_file.name)

else:
    st.warning("Please upload an MRI image to proceed.")


