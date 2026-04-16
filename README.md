# EmotionSense — Multi-Class Emotion Detection

A complete end-to-end MLOps project that detects 6 emotions in text using a **custom neural network built from scratch in NumPy**, trained on the real [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset.

## Emotions Detected
`sadness` · `joy` · `love` · `anger` · `fear` · `surprise`

## Dataset
- **Source:** dair-ai/emotion (Saravia et al., EMNLP 2018)
- **Size:** 20,000 real English tweets
- **HuggingFace:** https://huggingface.co/datasets/dair-ai/emotion

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download real dataset from HuggingFace
python scripts/download_dataset.py

# 3. Train the custom MLP
python train.py

# 4. Start API
uvicorn main:app --reload

# 5. Open frontend
open frontend/index.html
```

## Run with Docker

```bash
docker build -t emotionsense .
docker run -p 8000:8000 emotionsense
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Detect emotion from text |
| GET | `/metrics` | Model performance metrics |
| GET | `/health` | Health check |

## Architecture
- **Model:** Custom MLP (NumPy only) — Input → 256 → 128 → 64 → 6 (Softmax)
- **Features:** He init, ReLU, Dropout, Momentum SGD, Early Stopping
- **Text Features:** TF-IDF (8,000 features, 1+2-grams)
- **API:** FastAPI + Pydantic
- **CI/CD:** GitHub Actions → Docker Hub
