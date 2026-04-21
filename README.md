# EmotionSense v4 — Multi-Class Emotion Detection

[![CI/CD](https://github.com/SOMESH19MISHRA/EmotionSense/actions/workflows/deploy.yml/badge.svg)](https://github.com/SOMESH19MISHRA/EmotionSense/actions)
![Accuracy](https://img.shields.io/badge/Accuracy-92.8%25-brightgreen)
![Model](https://img.shields.io/badge/Model-DistilBERT%20Fine--tuned-purple)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20Bedrock%20%7C%20EC2-orange)

> **92.8% accuracy** on 2,000 unseen tweets using fully fine-tuned DistilBERT.  
> Deployed on AWS EC2 · Model stored on AWS S3 · Compared against AWS Bedrock Claude 3 Haiku.

---

## 👤 Author

| Field | Value |
|---|---|
| **Student** | Somesh Mishra |
| **Enrollment** | E23CSEU1682 |
| **Teacher** | Mr. Naveen Kumar |
| **Course** | CSET-363 — AWS Cloud Support Associate |
| **Dataset** | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) |
| **GitHub** | [SOMESH19MISHRA/EmotionSense](https://github.com/SOMESH19MISHRA/EmotionSense) |

---

## 🎭 Emotions Detected (6 classes)

`joy` · `sadness` · `anger` · `fear` · `love` · `surprise`

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | **92.8%** |
| **Macro F1** | **88.9%** |
| **Precision** | **87.5%** |
| **Recall** | **90.6%** |
| vs AWS Bedrock | **+38.7%** |

---

## ☁️ AWS Services Used

| Service | Usage |
|---|---|
| **AWS S3** | Stores 254MB `best_model.pt`. API downloads it automatically at startup. |
| **AWS Bedrock** | Claude 3 Haiku zero-shot comparison via boto3. Benchmark: 54.05% accuracy. |
| **AWS EC2** | t2.medium Ubuntu 24.04. Hosts Docker container. Port 8000 public. |

---

## 🧠 Model Architecture

```
Text
 → DistilBERT (66M params, FULLY fine-tuned, unfrozen)
 → Mean Pooling (768-dim)
 → Dropout(0.3) → Dense(256, GELU) → Dropout(0.2) → Dense(6, Softmax)
 → Emotion + Confidence
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/SOMESH19MISHRA/EmotionSense.git
cd EmotionSense
pip install -r requirements.txt
python scripts/download_dataset.py
python -m uvicorn main:app --reload
# Open frontend/index.html in browser
```

---

## 🐳 Docker

```bash
docker build -t emotionsense .
docker run -p 8000:8000 emotionsense
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status + AWS services |
| POST | `/predict` | Emotion detection (our model) |
| POST | `/predict/bedrock` | AWS Bedrock Claude comparison |
| GET | `/metrics` | Full performance metrics |
| GET | `/aws` | All 3 AWS services documented |
| GET | `/health` | Health check |
