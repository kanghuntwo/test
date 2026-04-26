<!-- Created: 2026-04-21 00:00:00 -->

# CLAUDE.md — Desktop Version

This file provides guidance for the **desktop version** of the handwritten digit recognizer.

## File Creation Rules

- 모든 새 파일의 상단에 생성 날짜와 시간을 주석으로 표시한다.
  - Python: `# Created: YYYY-MM-DD HH:MM:SS`
  - JS/TS: `// Created: YYYY-MM-DD HH:MM:SS`
  - HTML: `<!-- Created: YYYY-MM-DD HH:MM:SS -->`
  - CSS: `/* Created: YYYY-MM-DD HH:MM:SS */`

## Overview

Tkinter 기반 데스크톱 GUI 앱. 캔버스에 숫자를 그리면 PyTorch 모델이 실시간으로 예측.

## Running the App

```bash
python digit_recognizer.py
```

Windows에서는 `DigitRecognizer.bat` 더블클릭으로도 실행 가능 (Python이 PATH에 있어야 함).

## Building the Standalone Executable

```bash
pyinstaller --onefile --windowed --clean --name "DigitRecognizer" --add-data "digit_cnn.pth;." digit_recognizer.py
# Output: dist/DigitRecognizer.exe
```

## Architecture

**Model (`DigitCNN`)** — PyTorch CNN: Conv×3 → MaxPool×2 → Dropout → FC×2 → 10 classes.
5 에포크 학습, MNIST 테스트 정확도 ~99%. 가중치는 `digit_cnn.pth`에 저장/로드.

**Inference (`preprocess_canvas`, `predict`)** — Tkinter 캔버스 PIL 이미지를 28×28 MNIST 텐서로 변환.
`transforms.ToTensor()` 미사용 — Pillow 인코더 버그(`NoneType has no attribute 'write'`) 우회.
MNIST 정규화: mean=0.1307, std=0.3081.

**GUI (`App`)** — 280×280 드로잉 캔버스(좌) + 예측 결과 패널(우).
예측 숫자, 신뢰도 %, 0–9 클래스 확률 바 표시. 모델 로딩은 백그라운드 스레드로 UI 블로킹 방지.
마우스 업 이벤트마다 자동 예측.

## Directory Structure

```
desktop_version/
├── digit_recognizer.py   # 메인 애플리케이션 (단일 파일)
├── digit_cnn.pth         # 학습된 모델 가중치
├── DigitRecognizer.bat   # Windows 실행 배치 파일
└── CLAUDE.md
```

## Key Decisions

- `digit_cnn.pth`가 없으면 MNIST 자동 다운로드 후 학습 실행.
- MNIST 데이터는 `./data/`에 저장.
- `preprocess_canvas`에서 PIL 이미지를 복사해 드로잉 이벤트와 추론 간 race condition 방지.
