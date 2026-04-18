import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report, confusion_matrix)
warnings.filterwarnings('ignore')

from model.hybrid_model import HybridEmotionClassifier

RANDOM_SEED = 42
MODEL_DIR   = 'model/saved'
DATA_PATH   = 'data/emotion_dataset.csv'
np.random.seed(RANDOM_SEED)
os.makedirs(MODEL_DIR, exist_ok=True)

print('=' * 60)
print('  EmotionSense v3 -- Hybrid Transformer + MLP')
print('=' * 60)

print('\nLoading dataset...')
df = pd.read_csv(DATA_PATH)
print(f'  {len(df)} samples | {df['emotion'].nunique()} classes')

le      = LabelEncoder()
le.fit(df['emotion'])
classes = list(le.classes_)

if 'split' in df.columns:
    train_df = df[df['split'] == 'train']
    val_df   = df[df['split'] == 'validation']
    test_df  = df[df['split'] == 'test']
else:
    from sklearn.model_selection import train_test_split
    y_all = le.transform(df['emotion'].values)
    tr, te = train_test_split(df, test_size=0.1, stratify=y_all, random_state=RANDOM_SEED)
    tr, va = train_test_split(tr, test_size=0.111, random_state=RANDOM_SEED)
    train_df, val_df, test_df = tr, va, te

y_train = le.transform(train_df['emotion'].values)
y_val   = le.transform(val_df['emotion'].values)
y_test  = le.transform(test_df['emotion'].values)

print(f'  Split: train={len(y_train)} val={len(y_val)} test={len(y_test)}')

hybrid = HybridEmotionClassifier(n_classes=6, n_heads=4, mlp_dims=[256,128,64])
hybrid.load_transformer()

EMB_CACHE = os.path.join(MODEL_DIR, 'embeddings_cache.npz')
if os.path.exists(EMB_CACHE):
    print('Loading cached embeddings...')
    cache = np.load(EMB_CACHE)
    X_train_emb = cache['train']
    X_val_emb   = cache['val']
    X_test_emb  = cache['test']
else:
    print('Encoding with DistilBERT...')
    t0 = time.time()
    X_train_emb = hybrid.encode(train_df['text'].values)
    X_val_emb   = hybrid.encode(val_df['text'].values)
    X_test_emb  = hybrid.encode(test_df['text'].values)
    print(f'  Done in {time.time()-t0:.1f}s')
    np.savez(EMB_CACHE, train=X_train_emb, val=X_val_emb, test=X_test_emb)

print(f'\nEmbedding shape: {X_train_emb.shape}')
print('\nTraining Custom Attention + GELU MLP...\n')
t0 = time.time()
hybrid.fit(X_train_emb, y_train, X_val_emb, y_val,
           epochs=40, batch_size=64, patience=7, verbose=True)
elapsed = time.time() - t0
print(f'\nDone in {elapsed:.1f}s')

y_pred = hybrid.predict(X_test_emb)
acc    = accuracy_score(y_test, y_pred)
f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
f1_wt  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
prec   = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec    = recall_score(y_test, y_pred, average='macro', zero_division=0)

print('=' * 55)
print(f'  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)')
print(f'  Macro F1       : {f1_mac:.4f}')
print(f'  Weighted F1    : {f1_wt:.4f}')
print(f'  Macro Precision: {prec:.4f}')
print(f'  Macro Recall   : {rec:.4f}')
print('=' * 55)
print()
print(classification_report(y_test, y_pred, target_names=classes))

joblib.dump(le, f'{MODEL_DIR}/label_encoder.pkl')
for i, (W, b) in enumerate(zip(hybrid.mlp.W, hybrid.mlp.b)):
    os.makedirs(f'{MODEL_DIR}/weights', exist_ok=True)
    np.save(f'{MODEL_DIR}/weights/W_{i}.npy', W)
    np.save(f'{MODEL_DIR}/weights/b_{i}.npy', b)

metrics = {
    'accuracy': round(acc,4), 'macro_f1': round(f1_mac,4),
    'weighted_f1': round(f1_wt,4), 'macro_precision': round(prec,4),
    'macro_recall': round(rec,4), 'training_time_sec': round(elapsed,1),
    'train_samples': len(y_train), 'val_samples': len(y_val),
    'test_samples': len(y_test), 'classes': classes,
    'model': 'DistilBERT + Custom Multi-Head Attention + Custom GELU MLP',
    'dataset': 'dair-ai/emotion -- 20,000 real English tweets (EMNLP 2018)',
    'history': {k:[round(v,4) for v in hybrid.mlp.history[k]] for k in hybrid.mlp.history},
    'per_class': {
        cls: {
            'precision': round(precision_score(y_test==i, y_pred==i, zero_division=0),4),
            'recall':    round(recall_score(y_test==i, y_pred==i, zero_division=0),4),
            'f1':        round(f1_score(y_test==i, y_pred==i, zero_division=0),4),
            'support':   int((y_test==i).sum()),
        } for i, cls in enumerate(classes)
    }
}
with open(f'{MODEL_DIR}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f'All saved to {MODEL_DIR}/')
