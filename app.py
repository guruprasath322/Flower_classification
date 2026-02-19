import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'flower_model.keras')

# Load the trained model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at {MODEL_PATH}")
        st.info("Please ensure 'flower_model.keras' is in the project directory.")
        st.stop()
    
    try:
        with st.spinner("Loading model..."):
            model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model = load_model()

# Class names (hardcoded based on dataset structure, or could be loaded if saved)
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

st.title("Flower Classification App 🌸")
st.write("Upload an image of a flower (Rose, Sunflower, Daisy, Tulip, Dandelion) and the model will predict its type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create batch axis
    img_array = img_array / 255.0 # Rescale

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = 100 * np.max(score)

    st.success(f"Prediction: **{predicted_class.title()}**")
    # st.info(f"Confidence: {confidence:.2f}%") # Optional: Display confidence
