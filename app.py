import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load the saved model
model = load_model('mnist_cnn_model.h5')

st.title("🧮 MNIST Digit Classifier")

st.write("Upload a clear 28x28 pixel handwritten digit image.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('L')  # 'L' = grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resize to 28x28
    image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(image)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Reshape to (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    st.write(f"🧾 **Predicted Digit:** {predicted_label}")
