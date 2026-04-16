"""
train.py
--------
Full training pipeline for EmotionSense.
Loads the real dair-ai/emotion dataset (download it first with
    python scripts/download_dataset.py
), trains a custom NumPy MLP, and saves all artifacts.

Run:
    python train.py
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score,
                              classification_report, confusion_matrix)

from model.preprocess import TextVectorizer
from model.neural_network import EmotionMLP

# ── Config ────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
DATA_PATH   = "data/emotion_dataset.csv"
MODEL_DIR   = "model/saved"
np.random.seed(RANDOM_SEED)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────

if not os.path.exists(DATA_PATH):
    print("Dataset not found. Running download script...")
    os.system("python scripts/download_dataset.py")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Total samples : {len(df)}")
print(f"  Classes       : {sorted(df['emotion'].unique())}")
print(f"\nClass distribution:")
print(df["emotion"].value_counts().to_string())

# ── Encode labels ─────────────────────────────────────────────────────────────

le = LabelEncoder()
y  = le.fit_transform(df["emotion"].values)
classes = list(le.classes_)
print(f"\nLabel encoding: {dict(zip(classes, le.transform(classes)))}")

# ── Use pre-split data if available, else split ───────────────────────────────

if "split" in df.columns:
    print("\nUsing original train/val/test splits from dataset...")
    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "validation"]
    test_df  = df[df["split"] == "test"]
    X_train_raw = train_df["text"].values
    X_val_raw   = val_df["text"].values
    X_test_raw  = test_df["text"].values
    y_train = le.transform(train_df["emotion"].values)
    y_val   = le.transform(val_df["emotion"].values)
    y_test  = le.transform(test_df["emotion"].values)
else:
    print("\nSplitting dataset 80/10/10 (stratified)...")
    X_tmp, X_test_raw, y_tmp, y_test = train_test_split(
        df["text"].values, y, test_size=0.10,
        random_state=RANDOM_SEED, stratify=y)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.111,
        random_state=RANDOM_SEED, stratify=y_tmp)

print(f"\nSplit sizes — train: {len(X_train_raw)}  val: {len(X_val_raw)}  test: {len(X_test_raw)}")

# ── TF-IDF Vectorisation ──────────────────────────────────────────────────────

print("\nVectorising text (TF-IDF, 8000 features, 1+2-grams)...")
vec     = TextVectorizer(max_features=8000, ngram_range=(1, 2))
X_train = vec.fit_transform(X_train_raw)
X_val   = vec.transform(X_val_raw)
X_test  = vec.transform(X_test_raw)
print(f"  Feature matrix: {X_train.shape}")

# ── Build custom MLP ──────────────────────────────────────────────────────────

print("\nBuilding custom NumPy MLP...")
model = EmotionMLP(
    input_dim    = vec.vocab_size,
    hidden_dims  = [256, 128, 64],
    output_dim   = len(classes),
    lr           = 0.05,
    momentum     = 0.9,
    lam          = 1e-4,
    dropout_rates= [0.3, 0.2, 0.0],
    lr_decay     = 0.97,
)
print(f"  Architecture: {vec.vocab_size} → 256 (ReLU, drop=0.3) → 128 (ReLU, drop=0.2) → 64 (ReLU) → {len(classes)} (Softmax)")

# ── Train ─────────────────────────────────────────────────────────────────────

print("\nTraining...\n")
t0 = time.time()
model.fit(X_train, y_train, X_val, y_val,
          epochs=60, batch_size=128, patience=8, verbose=True)
elapsed = time.time() - t0
print(f"\nTraining complete in {elapsed:.1f}s")

# ── Evaluate ──────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)

acc      = accuracy_score(y_test, y_pred)
f1_mac   = f1_score(y_test, y_pred, average="macro")
f1_wt    = f1_score(y_test, y_pred, average="weighted")
prec     = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec      = recall_score(y_test, y_pred, average="macro", zero_division=0)

print("\n" + "="*58)
print(f"  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Macro F1-Score    : {f1_mac:.4f}")
print(f"  Weighted F1-Score : {f1_wt:.4f}")
print(f"  Macro Precision   : {prec:.4f}")
print(f"  Macro Recall      : {rec:.4f}")
print("="*58)
print("\nPer-class Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=classes))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Save artifacts ────────────────────────────────────────────────────────────

joblib.dump(model, f"{MODEL_DIR}/mlp_model.pkl")
joblib.dump(vec,   f"{MODEL_DIR}/vectorizer.pkl")
joblib.dump(le,    f"{MODEL_DIR}/label_encoder.pkl")

metrics = {
    "accuracy": round(acc,4), "macro_f1": round(f1_mac,4),
    "weighted_f1": round(f1_wt,4), "macro_precision": round(prec,4),
    "macro_recall": round(rec,4), "training_time_sec": round(elapsed,1),
    "train_samples": len(X_train_raw), "val_samples": len(X_val_raw),
    "test_samples":  len(X_test_raw),  "classes": classes,
    "dataset": "dair-ai/emotion (HuggingFace) — 20,000 real English tweets",
    "history": {k: [round(v,4) for v in model.history[k]] for k in model.history},
    "per_class": {
        cls: {
            "precision": round(precision_score(y_test==i, y_pred==i, zero_division=0), 4),
            "recall":    round(recall_score(y_test==i, y_pred==i, zero_division=0), 4),
            "f1":        round(f1_score(y_test==i, y_pred==i, zero_division=0), 4),
            "support":   int((y_test==i).sum()),
        }
        for i, cls in enumerate(classes)
    }
}

with open(f"{MODEL_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nAll artifacts saved to {MODEL_DIR}/")
print("  mlp_model.pkl | vectorizer.pkl | label_encoder.pkl | metrics.json")
