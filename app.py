import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

# 🔒 Load model from the same folder where app.py is located
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.h5")
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# ✅ List of class names (should match your model)
class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
               'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy', 'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
               'Background_without_leaves']

# 🧪 Remedies mapping
remedies = {
    "Tomato___Early_blight": "Use fungicides like Mancozeb; remove infected leaves.",
    "Tomato___Late_blight": "Apply copper-based fungicides; destroy infected debris.",
    "Tomato___Leaf_Mold": "Use sulfur sprays; improve air circulation.",
    "Tomato___healthy": "Your plant is healthy! Keep up the good work.",
    "Pepper__bell___healthy": "No issues detected.",
    "Potato___Late_blight": "Apply chlorothalonil; avoid overhead irrigation.",
    "Background_without_leaves": "This is not a crop leaf. Please upload a clear image of a crop leaf.",
    # Add more remedies here
}

# 🌱 Streamlit UI
st.set_page_config(page_title="Crop Disease Detector", layout="centered")
st.title("🌿 Crop Disease Detector + Remedy Suggestion")
st.write("Upload a leaf image of your crop to detect the disease and get remedy suggestions.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the disease
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    st.success(f"✅ Predicted Disease: **{predicted_class}**")

    # Show remedy
    remedy = remedies.get(predicted_class, "No remedy found for this disease class.")
    st.info(f"💊 Suggested Remedy: {remedy}")
