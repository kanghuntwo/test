# Created: 2026-04-21 00:00:00

import os
import io
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MODEL_PATH = os.path.join(os.path.dirname(__file__), "digit_cnn.pth")
IMG_SIZE = 28
_MEAN = 0.1307
_STD  = 0.3081


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_model(status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((_MEAN,), (_STD,)),
    ])

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    log("Downloading MNIST dataset...")
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    model     = DigitCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                preds = model(images.to(device)).argmax(dim=1)
                correct += (preds == labels.to(device)).sum().item()
        acc = correct / len(test_ds) * 100
        log(f"Epoch {epoch}/5  loss={total_loss/len(train_loader):.4f}  acc={acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    log(f"Model saved → {MODEL_PATH}")
    return model


def load_or_train_model():
    model = DigitCNN()
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    else:
        model = train_model()
    model.eval()
    return model


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """PNG/JPEG bytes (from browser canvas) → 1×1×28×28 tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    # Composite onto white background to flatten alpha
    background = Image.new("RGB", img.size, "white")
    background.paste(img, mask=img.split()[3])
    gray = background.convert("L")
    gray = ImageOps.invert(gray)
    gray = gray.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(gray, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def predict(model: DigitCNN, tensor: torch.Tensor):
    """Return (digit, confidence_percent, probs_list)."""
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    digit = int(probs.argmax())
    return digit, float(probs[digit]) * 100, probs.tolist()
