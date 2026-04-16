"""
download_dataset.py
-------------------
Downloads the real dair-ai/emotion dataset from HuggingFace.
Source : https://huggingface.co/datasets/dair-ai/emotion
Paper  : CARER (EMNLP 2018) — Saravia et al.
License: For educational and research purposes only.

Dataset: 20,000 English tweets labelled with 6 emotions:
  sadness, joy, love, anger, fear, surprise

Run:
    python scripts/download_dataset.py

Output:
    data/emotion_dataset.csv   (combined train + val + test, 20,000 rows)
"""

import os, sys, urllib.request, json
import pandas as pd

PARQUET_URLS = {
    "train":      "https://huggingface.co/datasets/dair-ai/emotion/resolve/refs%2Fconvert%2Fparquet/split/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/dair-ai/emotion/resolve/refs%2Fconvert%2Fparquet/split/validation/0000.parquet",
    "test":       "https://huggingface.co/datasets/dair-ai/emotion/resolve/refs%2Fconvert%2Fparquet/split/test/0000.parquet",
}

LABEL_MAP = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

OUTPUT = "data/emotion_dataset.csv"
CACHE  = "data/raw"


def download_parquet(split: str, url: str) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{split}.parquet")
    if not os.path.exists(path):
        print(f"  Downloading {split} split...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"  ERROR: Could not download {split}: {e}")
            print("  Make sure you have internet access and try again.")
            sys.exit(1)
    else:
        print(f"  {split} already cached.")
    df = pd.read_parquet(path)
    return df


def main():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(OUTPUT):
        df = pd.read_csv(OUTPUT)
        print(f"Dataset already exists: {OUTPUT} ({len(df)} rows)")
        return

    print("Downloading dair-ai/emotion dataset from HuggingFace...\n")
    frames = []
    for split, url in PARQUET_URLS.items():
        df = download_parquet(split, url)
        df["split"] = split
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Map integer labels to string emotion names
    combined["label"] = combined["label"].map(LABEL_MAP)
    combined = combined.rename(columns={"label": "emotion"})
    combined = combined[["text", "emotion", "split"]]

    combined.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(combined)} samples → {OUTPUT}")
    print("\nClass distribution:")
    print(combined["emotion"].value_counts().to_string())
    print(f"\nSplit distribution:")
    print(combined["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
