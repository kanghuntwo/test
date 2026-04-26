<!-- Created: 2026-04-21 00:00:00 -->

# CLAUDE.md — Web Version

This file provides guidance for the **web version** of the handwritten digit recognizer.

## File Creation Rules

- 모든 새 파일의 상단에 생성 날짜와 시간을 주석으로 표시한다.
  - JS/TS: `// Created: YYYY-MM-DD HH:MM:SS`
  - HTML: `<!-- Created: YYYY-MM-DD HH:MM:SS -->`
  - CSS: `/* Created: YYYY-MM-DD HH:MM:SS */`
  - Python: `# Created: YYYY-MM-DD HH:MM:SS`

## Overview

브라우저에서 손글씨 숫자를 그리면 실시간으로 인식하는 웹 애플리케이션.

## Planned Stack

- **Frontend** — HTML5 Canvas + Vanilla JS (또는 React)
- **Backend** — Python FastAPI (REST API)
- **Model** — ONNX 또는 TorchServe로 export한 `digit_cnn` 모델

## Directory Structure (planned)

```
web_version/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── backend/
│   ├── main.py          # FastAPI app
│   ├── model.py         # 모델 로드 및 추론
│   └── requirements.txt
└── CLAUDE.md
```

## Running the App

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend — open in browser
open frontend/index.html
```

## Key Decisions

- Canvas 드로잉은 28×28로 다운샘플링 후 MNIST 정규화(mean=0.1307, std=0.3081) 적용.
- 모델 파일(`digit_cnn.pth` 또는 `.onnx`)은 backend/ 폴더에 위치.
- CORS 설정 필요 (FastAPI `CORSMiddleware`).
