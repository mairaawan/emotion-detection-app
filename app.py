# ================== APP ==================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import os
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Emotion Recognition", layout="wide")

# ================= CONFIG =================
MODEL_PATH = "emotion_model.pth"
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ================= MODEL =================
class FER_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*3*3, 256), nn.ReLU(),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = FER_CNN().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        st.success("✅ Trained model loaded successfully!")
    else:
        st.warning("⚠️ No trained model found. Please train first.")
    return model

model = load_model()

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ================= SIDEBAR =================
st.sidebar.title("🎯 Emotion Recognition")
page = st.sidebar.radio("Navigation", ["Project Info", "Dataset Preview", "Live Camera Prediction"])

# ================= PROJECT INFO =================
if page == "Project Info":
    st.title("Emotion Recognition System")
    st.markdown("""
        This interactive app demonstrates a **Facial Emotion Recognition** system using **PyTorch** and **CNNs**.
        - **Classes:** angry, disgust, fear, happy, neutral, sad, surprise
        - **Model:** Custom CNN trained on FER dataset
        - **Real-time predictions:** Using your webcam
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/3f/Emotions_wheel.png", caption="Emotions Overview", use_column_width=True)

# ================= DATASET PREVIEW =================
elif page == "Dataset Preview":
    st.title("📂 Dataset Samples")
    dataset_path = "dataset/train"
    if not os.path.exists(dataset_path):
        st.warning("Dataset not found! Please unzip your dataset into 'dataset/train'.")
    else:
        cols = st.columns(len(CLASS_NAMES))
        for i, cls in enumerate(CLASS_NAMES):
            cls_path = os.path.join(dataset_path, cls)
            if os.path.exists(cls_path):
                img_name = os.listdir(cls_path)[0]
                img = Image.open(os.path.join(cls_path, img_name))
                cols[i].image(img, caption=cls, use_column_width=True)

# ================= LIVE CAMERA =================
elif page == "Live Camera Prediction":
    st.title("📷 Real-time Emotion Prediction")
    st.markdown("Allow camera access and see the predicted emotion live!")

    run = st.checkbox("Run Webcam")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Unable to access webcam")
            break

        # Convert to grayscale and resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (48,48))
        face = transform(Image.fromarray(face)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(face)
            pred = CLASS_NAMES[output.argmax(1).item()]

        # Display prediction on frame
        cv2.putText(frame, f"Prediction: {pred}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

    cap.release()
