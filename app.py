import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Emotion Detection System",
    page_icon="😊",
    layout="wide"
)

# ================= TITLE =================
st.markdown(
    "<h1 style='text-align: center;'>🎭 Facial Emotion Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Deep Learning based Real-Time Emotion Recognition</p>",
    unsafe_allow_html=True
)

# ================= CONFIG =================
MODEL_PATH = "emotion_model.pth"
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
NUM_CLASSES = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= MODEL =================
class FER_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ================= LOAD MODEL SAFELY =================
model = FER_CNN().to(device)
model_loaded = False

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model_loaded = True

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ================= SIDEBAR =================
st.sidebar.title("📌 Project Info")
st.sidebar.info(
    """
**Project:** Emotion Detection  
**Model:** CNN (PyTorch)  
**Input:** Camera Image  
**Classes:** 7 Emotions  
"""
)

if model_loaded:
    st.sidebar.success("✅ Model Loaded")
else:
    st.sidebar.warning("⚠️ Model Not Found")

# ================= MAIN UI =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Capture Image")
    camera_input = st.camera_input("Take a photo")

with col2:
    st.subheader("📊 Prediction")

    if camera_input is not None:
        image = Image.open(camera_input).convert("L")
        st.image(image, caption="Captured Image", width=250)

        if not model_loaded:
            st.error("❌ Model file not found (`emotion_model.pth`)")
        else:
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            pred_index = np.argmax(probs)
            pred_label = CLASSES[pred_index]
            confidence = probs[pred_index] * 100

            st.markdown(
                f"""
                <h3 style='color: green;'>Prediction: {pred_label}</h3>
                <h4>Confidence: {confidence:.2f}%</h4>
                """,
                unsafe_allow_html=True
            )

            # Probability bars
            st.subheader("Emotion Probabilities")
            for i, emotion in enumerate(CLASSES):
                st.progress(float(probs[i]))
                st.write(f"{emotion}: {probs[i]*100:.2f}%")

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>© 2026 Emotion Detection System | Built with Streamlit & PyTorch</p>",
    unsafe_allow_html=True
)
