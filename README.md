# 🧮 MNIST Handwritten Digit Classifier Web App

An AI-powered web app that uses a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9).  
Upload a clear image of a digit — the app predicts which digit it is, instantly!

---

## ✅ **What this project does**

- Loads a trained CNN model (`mnist_cnn_model.h5`) that recognizes digits based on the MNIST dataset.
- Lets a user upload a **28×28 pixel grayscale image** of a digit.
- Preprocesses the image: converts to grayscale, resizes, normalizes.
- Runs the CNN to predict the digit.
- Displays the prediction interactively in a Streamlit web interface.

---

## 🔍 **How it works — Detailed**

1️⃣ **Dataset:**  
   - Uses the classic **MNIST** dataset of handwritten digits (70,000 images of digits 0–9).

2️⃣ **Model:**  
   - Built using **TensorFlow/Keras**
   - Architecture:
     - 2 Convolutional layers to extract features.
     - MaxPooling layers to downsample.
     - Flatten + Dense layers for classification.
   - Trained for multiple epochs for high accuracy.
   - Saved as `mnist_cnn_model.h5`.

3️⃣ **Web App:**  
   - Built with **Streamlit**.
   - Users upload an image.
   - The app:
     - Converts the image to grayscale.
     - Resizes it to 28×28 pixels.
     - Normalizes pixel values (0–1).
     - Predicts the digit using the loaded model.
   - Shows the uploaded image and the predicted digit.

---

## ⚙️ **Tech Stack**

| Tool | Purpose |
|------|----------|
| Python | Main programming language |
| TensorFlow | Deep learning framework for the CNN |
| Streamlit | Turns Python code into an interactive web app |
| NumPy | Efficient numerical operations |
| PIL (Pillow) | Image processing |

---

## 🚀 **How to Run Locally**

Follow these clear steps to run it on your own machine:

### 📌 1️⃣ Clone the repository
```bash
git clone https://github.com/ogrk/MNIST-Digit-Classifier.git
cd MNIST-Digit-Classifier
