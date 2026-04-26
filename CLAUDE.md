# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## File Creation Rules

- **모든 새 파일의 상단에 생성 날짜와 시간을 주석으로 표시한다.** 언어에 맞는 주석 형식을 사용한다.
  - Python: `# Created: YYYY-MM-DD HH:MM:SS`
  - JS/TS: `// Created: YYYY-MM-DD HH:MM:SS`
  - HTML: `<!-- Created: YYYY-MM-DD HH:MM:SS -->`
  - CSS: `/* Created: YYYY-MM-DD HH:MM:SS */`

## Running the App

```bash
python digit_recognizer.py
```

Or double-click `DigitRecognizer.bat` in Windows Explorer (requires Python on PATH).

## Building the Standalone Executable

```bash
pyinstaller --onefile --windowed --clean --name "DigitRecognizer" --add-data "digit_cnn.pth;." digit_recognizer.py
# Output: dist/DigitRecognizer.exe
```

## Architecture

`digit_recognizer.py` is a single-file app with three layers:

**Model (`DigitCNN`)** — PyTorch CNN: Conv×3 → MaxPool×2 → Dropout → FC×2 → 10 classes. Trained on MNIST (5 epochs, ~99% test accuracy). Weights saved to / loaded from `digit_cnn.pth`.

**Inference (`preprocess_canvas`, `predict`)** — Converts the Tkinter canvas PIL image to a 28×28 MNIST-style tensor via numpy (not `transforms.ToTensor()` — bypasses a Pillow encoder bug that causes `NoneType has no attribute 'write'`). Normalises with MNIST mean/std (0.1307 / 0.3081).

**GUI (`App`)** — Tkinter window with a 280×280 drawing canvas (left) and a results panel (right) showing the predicted digit, confidence %, and a probability bar for each class 0–9. Model loading runs in a background thread so the UI stays responsive. Auto-predicts on every mouse-up event.

## Key Decisions

- `digit_cnn.pth` must exist alongside the script (or be embedded via `--add-data`) for instant startup; if absent, MNIST is downloaded and training runs automatically.
- MNIST data is downloaded to `./data/` on first run via `torchvision.datasets.MNIST`.
- `preprocess_canvas` copies the PIL image before processing to avoid race conditions between draw events and inference.
