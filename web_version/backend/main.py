# Created: 2026-04-21 00:00:00

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from model import load_or_train_model, preprocess_image, predict

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["net"] = load_or_train_model()
    yield
    ml_model.clear()


app = FastAPI(title="Digit Recognizer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))


@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    tensor = preprocess_image(image_bytes)
    digit, confidence, probs = predict(ml_model["net"], tensor)

    return {
        "digit": digit,
        "confidence": round(confidence, 2),
        "probabilities": [round(p * 100, 2) for p in probs],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "net" in ml_model}
