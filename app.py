# ================= IMPORTS =================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import os

# ================= CONFIG =================
MODEL_PATH = "emotion_model.pth"   # your trained model
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ================= MODEL =================
class FER_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),
            nn.Linear(512, NUM_CLASSES)
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

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Emotion Detection App", layout="wide")
st.title("😊 Real-Time Emotion Detection")
st.markdown("This app detects facial emotions from your camera in real-time using a trained CNN model.")

# Sidebar
st.sidebar.header("Settings")
run_camera = st.sidebar.checkbox("Enable Camera", value=False)
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png"])

# ================= HELPER FUNCTIONS =================
def predict_emotion(image, model):
    img = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img)
        prob = nn.Softmax(dim=1)(output)
        _, pred = torch.max(output, 1)
    return EMOTIONS[pred.item()], prob.cpu().numpy()[0]

# ================= IMAGE UPLOAD =================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    emotion, prob = predict_emotion(image, model)
    st.write(f"**Predicted Emotion:** {emotion}")
    st.bar_chart({EMOTIONS[i]: float(prob[i]) for i in range(NUM_CLASSES)})

# ================= CAMERA =================
if run_camera:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    st.write("Press 'q' to stop camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access camera.")
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Predict emotion
        emotion, prob = predict_emotion(pil_image, model)

        # Display on frame
        cv2.putText(frame, f"{emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # Stop condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
