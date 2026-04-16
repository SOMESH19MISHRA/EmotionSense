"""
main.py  —  EmotionSense FastAPI backend
Run: uvicorn main:app --reload
"""
import os, json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_DIR = "model/saved"
model   = joblib.load(f"{MODEL_DIR}/mlp_model.pkl")
vec     = joblib.load(f"{MODEL_DIR}/vectorizer.pkl")
le      = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
with open(f"{MODEL_DIR}/metrics.json") as f:
    METRICS = json.load(f)

EMOTION_META = {
    "joy":      {"emoji": "😄", "color": "#F59E0B", "description": "Happiness, delight, and positivity"},
    "sadness":  {"emoji": "😢", "color": "#3B82F6", "description": "Grief, sorrow, and melancholy"},
    "anger":    {"emoji": "😡", "color": "#EF4444", "description": "Frustration, outrage, and fury"},
    "fear":     {"emoji": "😨", "color": "#8B5CF6", "description": "Anxiety, dread, and apprehension"},
    "love":     {"emoji": "❤️",  "color": "#EC4899", "description": "Affection, warmth, and care"},
    "surprise": {"emoji": "😲", "color": "#F97316", "description": "Shock, amazement, and disbelief"},
}

app = FastAPI(title="EmotionSense API", version="1.0.0",
              description="6-class emotion detection using a custom NumPy MLP.")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "EmotionSense API is running", "version": "1.0.0"}

@app.post("/predict")
def predict(body: TextInput):
    if not body.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    X     = vec.transform([body.text])
    proba = model.predict_proba(X)[0]
    idx   = int(np.argmax(proba))
    label = le.classes_[idx]
    meta  = EMOTION_META.get(label, {})
    return {
        "emotion":     label,
        "confidence":  round(float(proba[idx]), 4),
        "emoji":       meta.get("emoji", ""),
        "color":       meta.get("color", "#888"),
        "description": meta.get("description", ""),
        "all_probabilities": {
            le.classes_[i]: round(float(proba[i]), 4)
            for i in range(len(le.classes_))
        },
    }

@app.get("/metrics")
def get_metrics():
    return {
        "accuracy":        METRICS["accuracy"],
        "macro_f1":        METRICS["macro_f1"],
        "macro_precision": METRICS["macro_precision"],
        "macro_recall":    METRICS["macro_recall"],
        "train_samples":   METRICS["train_samples"],
        "dataset":         METRICS.get("dataset", "dair-ai/emotion"),
        "classes":         METRICS["classes"],
        "per_class":       METRICS["per_class"],
    }

@app.get("/health")
def health():
    return {"status": "ok"}
