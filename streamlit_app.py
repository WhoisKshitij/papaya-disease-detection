import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Load your trained CNN model
model = load_model("papaya_cnn_model.h5")

# Set up the Streamlit app layout
st.set_page_config(page_title="Papaya Disease Detection", layout="centered")

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #2C3E50;
    }
    .footer {
        text-align: center;
        font-size: 12px;
        color: #888;
        margin-top: 30px;
    }
    </style>
    <div class="title">🍃 Papaya Fruit Disease Detection using CNN</div>
""", unsafe_allow_html=True)

st.write("Upload a papaya fruit image to check if it is **Diseased** or **Healthy**.")

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]  # Drop alpha channel if present
    return np.expand_dims(img_array, axis=0)

# File uploader
uploaded_file = st.file_uploader("Choose a papaya fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Processing and classifying...")

    # Preprocess and predict
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)[0][0]
    predicted_class = "Healthy" if prediction > 0.5 else "Diseased"
    confidence = prediction if predicted_class == "Healthy" else 1 - prediction

    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    # Probability bar graph
    st.subheader("Prediction Probability")
    fig, ax = plt.subplots()
    ax.bar(["Diseased", "Healthy"], [1 - prediction, prediction], color=["red", "green"])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Confidence")
    for i, v in enumerate([1 - prediction, prediction]):
        ax.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')
    st.pyplot(fig)

# Footer
st.markdown("""
    <div class="footer">
        Developed by 
        <a href="mailto:kshitijsukhdeve@gmail.com">kshitijsukhdeve@gmail.com</a>,
        <a href="mailto:kuldeepjaiswal108@gmail.com">kuldeepjaiswal108@gmail.com</a>,
        <a href="mailto:manseesahu091203@gmail.com">manseesahu091203@gmail.com</a>,
        <a href="mailto:devikagupta592@gmail.com">devikagupta592@gmail.com</a>
    </div>
""", unsafe_allow_html=True)
