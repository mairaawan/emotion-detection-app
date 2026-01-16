# ================= IMPORTS =================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# ================= CONFIG =================
MODEL_PATH = "emotion_model.pth"
BATCH_SIZE = 64
LR = 5e-4
EPOCHS = 50
NUM_CLASSES = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ================= TRANSFORMS =================
train_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

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

# ================= MAIN =================
def main():

    # ---- Load Dataset ----
    train_dataset = datasets.ImageFolder("dataset/train", transform=train_transform)
    val_dataset = datasets.ImageFolder("dataset/test", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    classes = train_dataset.classes
    print("Classes:", classes)

    # ---- Class Weights ----
    targets = [y for _, y in train_dataset.samples]
    class_count = Counter(targets)
    weights = torch.tensor([1.0 / class_count[i] for i in range(NUM_CLASSES)]).to(device)

    # ---- Model ----
    model = FER_CNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---- Check if model exists ----
    if os.path.exists(MODEL_PATH):
        print("📦 Loading trained model...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("✅ Model loaded")
        TRAIN_MODEL = False
    else:
        TRAIN_MODEL = True

    # ================= TRAIN =================
    if TRAIN_MODEL:
        print("\n🚀 Training started...\n")
        for epoch in range(EPOCHS):
            model.train()
            correct, total, loss_sum = 0, 0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)
            acc = correct / total
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Acc: {acc:.4f}")

        # ---- Save Model ----
        torch.save(model.state_dict(), MODEL_PATH)
        print("\n✅ Model saved successfully")

    # ================= EVALUATION =================
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            outputs = model(x)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n🎯 Test Accuracy: {acc:.4f}")

    # ================= CONFUSION MATRIX =================
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# ================= RUN =================
if __name__ == "__main__":
    main()
