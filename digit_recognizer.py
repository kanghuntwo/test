"""
Handwritten Digit Recognizer
- Trains a CNN on MNIST dataset using PyTorch
- Provides a Tkinter GUI canvas where users draw digits with the mouse
- Predicts the drawn digit in real time
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms  # used only in train_model

MODEL_PATH = "digit_cnn.pth"
CANVAS_SIZE = 280   # drawing canvas (pixel)
IMG_SIZE    = 28    # MNIST image size


# ── Model ──────────────────────────────────────────────────────────────────────

class DigitCNN(nn.Module):
    """Small CNN that matches MNIST input (1×28×28) → 10 classes."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # → 64×14×14
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # → 128×7×7
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


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(status_callback=None):
    """Download MNIST and train the CNN; returns the trained model."""

    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    log("Downloading MNIST dataset...")
    train_ds = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    model     = DigitCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    EPOCHS = 5
    for epoch in range(1, EPOCHS + 1):
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

        # Quick validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                preds = model(images.to(device)).argmax(dim=1)
                correct += (preds == labels.to(device)).sum().item()
        acc = correct / len(test_ds) * 100
        log(f"Epoch {epoch}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  acc={acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    log(f"Model saved → {MODEL_PATH}")
    return model


def load_or_train_model(status_callback=None):
    model = DigitCNN()
    if os.path.exists(MODEL_PATH):
        if status_callback:
            status_callback("Loading saved model...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    else:
        model = train_model(status_callback)
    model.eval()
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

_MEAN = 0.1307
_STD  = 0.3081

def preprocess_canvas(pil_img: Image.Image) -> torch.Tensor:
    """Resize canvas snapshot to 28×28 MNIST-style tensor.

    Bypasses PIL's internal encoder (tobytes path) by converting directly to a
    numpy array — avoids the 'NoneType has no attribute write' encoder bug that
    appears in newer Pillow builds with transforms.ToTensor().
    """
    # Work on an explicit copy so clearing the canvas mid-call is safe
    snapshot = pil_img.copy()
    gray = snapshot.convert("L")
    # Invert: canvas is white background, MNIST is black background
    gray = ImageOps.invert(gray)
    gray = gray.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    # numpy path: uint8 → float32 [0,1] → normalise → tensor
    arr = np.array(gray, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # 1×1×28×28
    return tensor


def predict(model: DigitCNN, tensor: torch.Tensor):
    """Return (digit, confidence_percent, all_probs array)."""
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    digit = int(probs.argmax())
    return digit, float(probs[digit]) * 100, probs


# ── GUI ────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    BRUSH_RADIUS = 14   # drawing brush radius in canvas pixels

    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.resizable(False, False)

        self.model: DigitCNN | None = None
        self._pil_img  = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self._pil_draw = ImageDraw.Draw(self._pil_img)

        self._build_ui()
        self._start_model_loading()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        # ── Left: drawing canvas ──
        left = tk.Frame(self)
        left.grid(row=0, column=0, **pad)

        tk.Label(left, text="Draw a digit (0–9) below:", font=("Arial", 11)).pack(anchor="w")

        self.canvas = tk.Canvas(
            left, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="white", cursor="crosshair",
            highlightthickness=2, highlightbackground="#555",
        )
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>",       self._on_draw)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        btn_frame = tk.Frame(left)
        btn_frame.pack(fill="x", pady=(6, 0))
        tk.Button(btn_frame, text="Clear",   width=10, command=self._clear,   bg="#e74c3c", fg="white").pack(side="left",  padx=4)
        tk.Button(btn_frame, text="Predict", width=10, command=self._predict, bg="#2ecc71", fg="white").pack(side="right", padx=4)

        # ── Right: result panel ──
        right = tk.Frame(self, width=220)
        right.grid(row=0, column=1, sticky="ns", **pad)

        tk.Label(right, text="Prediction", font=("Arial", 12, "bold")).pack(pady=(4, 0))

        self.result_var = tk.StringVar(value="—")
        tk.Label(right, textvariable=self.result_var, font=("Arial", 64, "bold"), fg="#2c3e50").pack()

        self.conf_var = tk.StringVar(value="Draw a digit and\npress Predict")
        tk.Label(right, textvariable=self.conf_var, font=("Arial", 10), justify="center").pack()

        tk.Label(right, text="All probabilities:", font=("Arial", 10, "bold")).pack(pady=(12, 2))

        self.bars: list[tuple[tk.Label, ttk.Progressbar, tk.Label]] = []
        for d in range(10):
            row = tk.Frame(right)
            row.pack(fill="x", padx=4, pady=1)
            lbl  = tk.Label(row, text=str(d), width=2, font=("Courier", 9, "bold"))
            bar  = ttk.Progressbar(row, length=120, maximum=100)
            pct  = tk.Label(row, text="  0%", width=5, font=("Courier", 9))
            lbl.pack(side="left")
            bar.pack(side="left", padx=2)
            pct.pack(side="left")
            self.bars.append((lbl, bar, pct))

        # ── Bottom: status bar ──
        self.status_var = tk.StringVar(value="Loading model…")
        tk.Label(self, textvariable=self.status_var, font=("Arial", 9),
                 fg="#555", anchor="w").grid(row=1, column=0, columnspan=2,
                                              sticky="ew", padx=8, pady=(0, 4))

    # ── Model loading (background thread) ─────────────────────────────────────

    def _start_model_loading(self):
        def worker():
            try:
                m = load_or_train_model(self._set_status)
                self.model = m
                self._set_status("Model ready — draw a digit and press Predict!")
            except Exception as exc:
                self._set_status(f"Error: {exc}")
        threading.Thread(target=worker, daemon=True).start()

    def _set_status(self, msg: str):
        self.after(0, lambda: self.status_var.set(msg))

    # ── Drawing callbacks ──────────────────────────────────────────────────────

    def _on_draw(self, event):
        r = self.BRUSH_RADIUS
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self._pil_draw.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def _on_release(self, _event):
        # Auto-predict on every stroke
        if self.model is not None:
            self._predict()

    # ── Actions ────────────────────────────────────────────────────────────────

    def _clear(self):
        self.canvas.delete("all")
        self._pil_img  = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self._pil_draw = ImageDraw.Draw(self._pil_img)
        self.result_var.set("—")
        self.conf_var.set("Draw a digit and\npress Predict")
        for _, bar, pct in self.bars:
            bar["value"] = 0
            pct.config(text="  0%")

    def _predict(self):
        if self.model is None:
            messagebox.showinfo("Please wait", "Model is still loading…")
            return

        tensor = preprocess_canvas(self._pil_img)
        digit, conf, probs = predict(self.model, tensor)

        self.result_var.set(str(digit))
        self.conf_var.set(f"Confidence: {conf:.1f}%")

        for d, (lbl, bar, pct) in enumerate(self.bars):
            v = float(probs[d]) * 100
            bar["value"] = v
            pct.config(text=f"{v:4.1f}%")
            lbl.config(fg="#e74c3c" if d == digit else "black")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
