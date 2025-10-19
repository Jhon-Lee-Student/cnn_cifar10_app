import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================
# APP TITLE & DESCRIPTION
# ============================
st.title("🧠 CIFAR-10 Image Classification")
st.write("""
Upload an image (32x32 pixels or larger) and let the trained CNN model predict its class.  
Model trained on the CIFAR-10 dataset with optimization (BatchNorm, Dropout, and Data Augmentation).
""")

# ============================
# LOAD MODEL
# ============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_cnn_optimized.h5")
    return model

model = load_model()

# CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ============================
# IMAGE UPLOAD SECTION
# ============================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    # ============================
    # DISPLAY RESULT
    # ============================
    st.subheader("Prediction Result")
    st.write(f"**Class:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Show probabilities
    st.bar_chart(predictions[0])

else:
    st.info("Please upload an image to start prediction.")
