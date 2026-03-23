import urllib
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ================= LOAD MODEL =================
MODEL_PATH = "Model/model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Get input size automatically (VERY IMPORTANT FIX)
INPUT_SIZE = model.input_shape[1:3]

# Labels (based on repo)
CLASS_NAMES = ["Apple", "Banana", "Mango", "Orange", "Pineapple"]

# ================= UI =================
st.set_page_config(page_title="Fruit Classifier", layout="centered")

st.markdown("<h1 style='text-align: center;'>🍎 Fruit Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Upload a fruit image</h3>", unsafe_allow_html=True)

image = None

# ================= INPUT =================
opt = st.selectbox(
    "Choose input method:",
    ("Select", "Upload from device", "Paste image URL"),
)

if opt == "Upload from device":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if file is not None:
        image = Image.open(file).convert("RGB")

elif opt == "Paste image URL":
    url = st.text_input("Enter Image URL")
    if url:
        try:
            image = Image.open(urllib.request.urlopen(url)).convert("RGB")
        except:
            st.error("Invalid URL")

# ================= PREDICTION =================
if image is not None:
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Classify"):
        # Resize using model's expected input size
        img = image.resize(INPUT_SIZE)

        # Convert to array
        img_array = np.array(img)

        # Normalize (IMPORTANT for this model)
        img_array = img_array / 255.0

        # Expand dims
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        result = CLASS_NAMES[class_index]

        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence:.2f}%")