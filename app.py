# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os, json, io
from typing import List

# -----------------------
# Lazy imports (won't crash if package missing)
# -----------------------
TF_AVAILABLE = False
ONNX_AVAILABLE = False
TFLITE_AVAILABLE = False
TF_IMPORT_ERROR = None
ort = None
tflite_runtime = None

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception as e:
    TF_IMPORT_ERROR = e
    tf = None

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except Exception:
    ort = None

# prefer tflite_runtime if available (smaller)
try:
    import tflite_runtime.interpreter as tflite_runtime
    TFLITE_AVAILABLE = True
except Exception:
    tflite_runtime = None
    # fallback to TF's lite interpreter if TF is available
    if TF_AVAILABLE:
        try:
            from tensorflow.lite import Interpreter as TF_LITE_INTERPRETER
            TFLITE_AVAILABLE = True
            tflite_runtime = TF_LITE_INTERPRETER
        except Exception:
            tflite_runtime = None

import streamlit as st
import numpy as np
from PIL import Image

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="fish_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Class names
CLASS_NAMES = ["Salmon", "Tuna", "Trout", "Mackerel", "Catfish"]

# Preprocess function
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# Prediction function
def predict(image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Streamlit UI
st.title("🐟 Fish Classification App")
uploaded_file = st.file_uploader("Upload a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(image)
    preds = predict(img_array)
    class_index = np.argmax(preds)
    confidence = np.max(preds) * 100

    st.success(f"Prediction: {CLASS_NAMES[class_index]} ({confidence:.2f}%)")

# -----------------------
# Helper: load class names
# -----------------------
def load_class_names() -> List[str]:
    if os.path.exists("class_names.txt"):
        with open("class_names.txt", "r", encoding="utf-8") as f:
            names = [l.strip() for l in f.readlines() if l.strip()]
            return names
    if os.path.exists("class_names.json"):
        with open("class_names.json", "r", encoding="utf-8") as f:
            return json.load(f)
    # default placeholder (user should replace)
    return ["class_0", "class_1", "class_2"]

CLASS_NAMES = load_class_names()

# -----------------------
# Detect which model files exist
# -----------------------
MODEL_H5 = "fish_model.h5"
MODEL_ONNX = "fish_model.onnx"
MODEL_TFLITE = "fish_model.tflite"

have_h5 = os.path.exists(MODEL_H5)
have_onnx = os.path.exists(MODEL_ONNX)
have_tflite = os.path.exists(MODEL_TFLITE)

# -----------------------
# Model loaders and predict wrappers
# -----------------------
keras_model = None
onnx_sess = None
tflite_interpreter = None
selected_backend = None

def try_load_keras():
    global keras_model
    if TF_AVAILABLE and have_h5:
        try:
            keras_model = tf.keras.models.load_model(MODEL_H5)
            return True
        except Exception as e:
            st.warning(f"Failed to load Keras model: {e}")
    return False

def try_load_onnx():
    global onnx_sess
    if ONNX_AVAILABLE and have_onnx:
        try:
            onnx_sess = ort.InferenceSession(MODEL_ONNX)
            return True
        except Exception as e:
            st.warning(f"Failed to load ONNX model: {e}")
    return False

def try_load_tflite():
    global tflite_interpreter
    if TFLITE_AVAILABLE and have_tflite:
        try:
            # tflite_runtime.Interpreter or TF's Interpreter both use same API
            tflite_interpreter = tflite_runtime.Interpreter(model_path=MODEL_TFLITE)
            tflite_interpreter.allocate_tensors()
            return True
        except Exception as e:
            st.warning(f"Failed to load TFLite model: {e}")
    return False

# Try backends in priority: Keras -> ONNX -> TFLite
if try_load_keras():
    selected_backend = "keras"
elif try_load_onnx():
    selected_backend = "onnx"
elif try_load_tflite():
    selected_backend = "tflite"
else:
    selected_backend = None

# -----------------------
# Preprocessing utility (auto-detect input shape if possible)
# -----------------------
def detect_input_size():
    # Keras
    if selected_backend == "keras" and keras_model is not None:
        try:
            shape = keras_model.input_shape
            # typical: (None, H, W, C)
            if isinstance(shape, tuple) and len(shape) >= 3:
                return (shape[1] or 224, shape[2] or 224)
        except Exception:
            pass
    # ONNX
    if selected_backend == "onnx" and onnx_sess is not None:
        try:
            inp = onnx_sess.get_inputs()[0]
            shp = inp.shape
            # shape might be [batch, ch, H, W] or [batch, H, W, ch]
            shp = [s if isinstance(s, int) else -1 for s in shp]
            if len(shp) == 4:
                # decide which is height,width by checking typical positions
                # try (N,H,W,C)
                for i in [(1,2), (2,3)]:
                    if shp[i[0]] > 0 and shp[i[1]] > 0:
                        return (shp[i[0]], shp[i[1]])
        except Exception:
            pass
    # TFLite
    if selected_backend == "tflite" and tflite_interpreter is not None:
        try:
            details = tflite_interpreter.get_input_details()[0]
            shp = details['shape']  # e.g. [1,224,224,3]
            if len(shp) == 4:
                return (int(shp[1]), int(shp[2]))
        except Exception:
            pass
    # fallback
    return (224, 224)

TARGET_SIZE = detect_input_size()

def preprocess_pil(img: Image.Image, target_size=TARGET_SIZE):
    img = img.convert("RGB")
    img = img.resize((target_size[1], target_size[0]))  # PIL uses (w,h)
    arr = np.array(img).astype(np.float32) / 255.0
    # Keras/TF typically expects NHWC
    # ONNX might expect NCHW — we'll handle inside predict if needed
    return arr

# -----------------------
# Prediction wrappers
# -----------------------
def predict_keras(img_arr: np.ndarray):
    x = np.expand_dims(img_arr, 0)  # NHWC
    preds = keras_model.predict(x)
    probs = preds[0]
    return probs

def predict_onnx(img_arr: np.ndarray):
    inp = onnx_sess.get_inputs()[0]
    name = inp.name
    # ONNX models often expect NCHW; try both forms
    x_nhwc = np.expand_dims(img_arr.astype(np.float32), 0)
    try:
        # try feeding NHWC first
        out = onnx_sess.run(None, {name: x_nhwc})
        probs = np.array(out[0])[0]
        return probs
    except Exception:
        # try NCHW
        x_nchw = np.transpose(x_nhwc, (0,3,1,2)).astype(np.float32)
        out = onnx_sess.run(None, {name: x_nchw})
        probs = np.array(out[0])[0]
        return probs

def predict_tflite(img_arr: np.ndarray):
    x = np.expand_dims(img_arr.astype(np.float32), 0)
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    idx = input_details[0]['index']
    tflite_interpreter.set_tensor(idx, x)
    tflite_interpreter.invoke()
    out = tflite_interpreter.get_tensor(output_details[0]['index'])
    probs = np.array(out)[0]
    return probs

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Fish Classifier (robust deploy)", page_icon="🐟", layout="centered")
st.title("🐟 Fish Classification (robust for Streamlit Cloud)")

st.markdown("This app auto-selects an available backend. It will not crash if `tensorflow` is missing. Upload an image and the app will use whichever model/runtime is available in the repo (h5 / onnx / tflite).")

st.write("**Detected:**")
backend_msg = selected_backend if selected_backend else "No supported model/runtime found"
st.info(f"Backend: **{backend_msg}**  \nTF installed: **{TF_AVAILABLE}**, ONNX runtime: **{ONNX_AVAILABLE}**, TFLite runtime: **{TFLITE_AVAILABLE}**")

# allow user to override labels if default placeholders
if CLASS_NAMES and CLASS_NAMES[0].startswith("class_"):
    st.warning("No `class_names.txt` detected. Please edit CLASS NAMES below (comma-separated) or upload a file named `class_names.txt` with one label per line.")
    txt = st.text_input("Enter class names (comma-separated)", value=",".join(CLASS_NAMES))
    if txt:
        CLASS_NAMES = [s.strip() for s in txt.split(",") if s.strip()]

uploaded_file = st.file_uploader("Upload fish image (jpg/png)", type=["jpg","jpeg","png"])
if uploaded_file is None:
    st.info("Waiting for image...")
else:
    img = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(img, caption="Uploaded image", use_column_width=True)
    arr = preprocess_pil(img, TARGET_SIZE)

    if selected_backend is None:
        st.error("No usable model loaded. Please do one of the following (choose one):\n\n"
                 "1. Deploy with a compatible Python + TensorFlow: set Streamlit Cloud Python to 3.11/3.12 (Advanced settings) and add `tensorflow` to `requirements.txt`, then upload `fish_model.h5`.\n\n"
                 "2. Convert your Keras model to ONNX or TFLite and upload `fish_model.onnx` or `fish_model.tflite` and add `onnxruntime` or `tflite-runtime` to requirements.\n\n"
                 "Conversion snippets (run locally/Colab) are shown below if you need them.")
        with st.expander("Show conversion snippets"):
            st.code("""# Convert Keras .h5 -> TFLite (run in Colab with TF available)
import tensorflow as tf
model = tf.keras.models.load_model('fish_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open('fish_model.tflite','wb').write(tflite_model)
""")
            st.code("""# Convert Keras .h5 -> ONNX (use tf2onnx in Colab)
!pip install -q tf2onnx
python -m tf2onnx.convert --keras fish_model.h5 --output fish_model.onnx --opset 13
""")
    else:
        with st.spinner("Predicting..."):
            try:
                if selected_backend == "keras":
                    probs = predict_keras(arr)
                elif selected_backend == "onnx":
                    probs = predict_onnx(arr)
                elif selected_backend == "tflite":
                    probs = predict_tflite(arr)
                else:
                    probs = None
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                probs = None

        if probs is not None:
            # Ensure CLASS_NAMES length matches probs
            n = len(probs)
            if len(CLASS_NAMES) != n:
                st.warning(f"class_names length ({len(CLASS_NAMES)}) doesn't match model output ({n}). Showing indices.")
                labels = [f"class_{i}" for i in range(n)]
            else:
                labels = CLASS_NAMES

            top_idx = int(np.argmax(probs))
            st.success(f"Prediction: **{labels[top_idx]}**  — confidence {100*probs[top_idx]:.2f}%")
            st.subheader("All class probabilities")
            for i, p in enumerate(probs):
                st.write(f"{labels[i]}: {100*p:.2f}%")
        else:
            st.error("No prediction was made.")

# Footer / helpful tips
st.markdown("---")
st.markdown("**Deployment tips:**\n\n"
            "- If you want to use the original `.h5` and TensorFlow on Streamlit Cloud, set your app to use **Python 3.11 or 3.12** in the app's *Advanced settings* before deploying, and include `tensorflow` in `requirements.txt` (but be aware TF is large). :contentReference[oaicite:3]{index=3}\n\n"
            "- To avoid the heavy TF install, convert your model to **ONNX** or **TFLite** in Colab and upload the converted model; then add `onnxruntime` or `tflite-runtime` to your `requirements.txt`. ONNX runtime is often easier to install and lighter than full TF. :contentReference[oaicite:4]{index=4}\n\n"
            "- The videos you referenced demonstrate the usual deployments with TF + Streamlit; they are useful but assume a TF-capable runtime. If Cloud gives Python 3.13 you must either change the runtime or use ONNX/TFLite. :contentReference[oaicite:5]{index=5}"
)
