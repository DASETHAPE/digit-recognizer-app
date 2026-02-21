import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="CNN Digit Recognizer", page_icon="🔢")
st.title("Handwritten Digit Recognizer 🧠")
st.write("Upload an image of a handwritten digit (0-9) and the AI will guess it!")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_cnn.h5')

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=150)
    
    # Preprocess the image to match the CNN's training data
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    
    img_array = np.array(image)
    img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    if st.button('Predict Digit'):
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        st.success(f"**Prediction: {predicted_class}**")
        st.info(f"Confidence: {confidence:.2f}%")
