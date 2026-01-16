# ================= IMPORTS =================
import streamlit as st
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# ================= CONFIG =================
MODEL_PATH = "emotion_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

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
            nn.Linear(512 * 3 * 3, 256), nn.ReLU(),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = FER_CNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ================= STREAMLIT APP =================
st.set_page_config(page_title="Emotion Detection", layout="wide")
st.title("🎭 Real-Time Emotion Detection")
st.markdown("""
This app uses a trained CNN to detect emotions from facial expressions in real-time.
""")

# Sidebar info
st.sidebar.header("About")
st.sidebar.markdown("""
- **Model:** CNN trained on FER dataset  
- **Classes:** angry, disgust, fear, happy, neutral, sad, surprise  
- **Author:** Your Name  
""")

# Select input mode
mode = st.radio("Choose input source:", ["Upload Image", "Use Webcam"])

# ================= IMAGE UPLOAD =================
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(image_tensor)
            pred = CLASSES[output.argmax(1).item()]
        st.success(f"Predicted Emotion: **{pred}**")

# ================= WEBCAM =================
elif mode == "Use Webcam":
    stframe = st.empty()
    run = st.button("Start Webcam")
    cap = cv2.VideoCapture(0)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not access webcam")
            break
        # Convert to PIL Image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Preprocess
        image_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(image_tensor)
            pred = CLASSES[output.argmax(1).item()]
        
        # Display prediction on frame
        cv2.putText(frame, f"Emotion: {pred}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,255,0), 2, cv2.LINE_AA)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    cap.release()
