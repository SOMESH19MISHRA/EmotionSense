"""
hybrid_model.py
---------------
Hybrid Emotion Classifier combining:
  1. DistilBERT Transformer  — contextual token embeddings
  2. Custom Multi-Head Self-Attention — built from scratch in NumPy
  3. Custom MLP classifier   — built from scratch in NumPy

Architecture:
  Text
   → DistilBERT tokenizer + encoder  (768-dim contextual embeddings)
   → Mean pooling over token embeddings
   → Custom Multi-Head Self-Attention (4 heads, from scratch)
   → Layer Normalisation (from scratch)
   → Dense(256, GELU) + Dropout(0.3)
   → Dense(128, GELU) + Dropout(0.2)
   → Dense(64,  GELU)
   → Dense(6,   Softmax)
   → Emotion label + confidence

Everything after DistilBERT is pure NumPy — no PyTorch layers used.
"""

import numpy as np


# ── Activations ───────────────────────────────────────────────────────────────

def gelu(x):
    """Gaussian Error Linear Unit — smoother than ReLU, used in transformers."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def gelu_grad(x):
    tanh_val = np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))
    sech2    = 1.0 - tanh_val**2
    dtanh    = np.sqrt(2.0 / np.pi) * (1.0 + 3 * 0.044715 * x**2)
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * dtanh

def softmax(Z):
    Z_s = Z - Z.max(axis=-1, keepdims=True)
    e   = np.exp(Z_s)
    return e / e.sum(axis=-1, keepdims=True)

def relu(Z):
    return np.maximum(0.0, Z)


# ── Layer Normalisation (from scratch) ────────────────────────────────────────

class LayerNorm:
    """Normalises across the feature dimension. Built from scratch."""
    def __init__(self, dim, eps=1e-6):
        self.gamma = np.ones(dim,  dtype=np.float32)
        self.beta  = np.zeros(dim, dtype=np.float32)
        self.eps   = eps
        self._cache = {}

    def forward(self, x):
        mu  = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1,  keepdims=True)
        xn  = (x - mu) / np.sqrt(var + self.eps)
        out = self.gamma * xn + self.beta
        self._cache = {"xn": xn, "var": var, "x": x, "mu": mu}
        return out

    def backward(self, dout):
        xn, var = self._cache["xn"], self._cache["var"]
        N  = dout.shape[-1]
        dgamma = (dout * xn).sum(axis=0)
        dbeta  = dout.sum(axis=0)
        dxn    = dout * self.gamma
        dvar   = (-0.5 * dxn * xn / (var + self.eps)).sum(axis=-1, keepdims=True)
        dmu    = (-dxn / np.sqrt(var + self.eps)).sum(axis=-1, keepdims=True)
        dx     = dxn / np.sqrt(var + self.eps) + \
                 2 * dvar * (self._cache["x"] - self._cache["mu"]) / N + \
                 dmu / N
        return dx, dgamma, dbeta


# ── Multi-Head Self-Attention (from scratch) ──────────────────────────────────

class MultiHeadAttention:
    """
    Multi-Head Self-Attention built entirely from scratch in NumPy.
    Given input X of shape (batch, seq_len, d_model):
      - Projects to Q, K, V via learned weight matrices
      - Splits into h heads
      - Computes scaled dot-product attention per head
      - Concatenates heads and projects back
    """

    def __init__(self, d_model=768, num_heads=4):
        assert d_model % num_heads == 0
        self.h      = num_heads
        self.d_k    = d_model // num_heads
        self.d_model= d_model
        scale       = np.sqrt(2.0 / d_model)

        # Query, Key, Value, Output projection weights
        self.W_Q = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_K = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_V = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_O = np.random.randn(d_model, d_model).astype(np.float32) * scale

        self.b_Q = np.zeros(d_model, dtype=np.float32)
        self.b_K = np.zeros(d_model, dtype=np.float32)
        self.b_V = np.zeros(d_model, dtype=np.float32)
        self.b_O = np.zeros(d_model, dtype=np.float32)

        self._cache = {}

    def _split_heads(self, x, batch):
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        x = x.reshape(batch, -1, self.h, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x, batch):
        """(batch, heads, seq, d_k) → (batch, seq, d_model)"""
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, -1, self.d_model)

    def forward(self, X, training=True):
        batch, seq, _ = X.shape

        Q = X @ self.W_Q + self.b_Q   # (batch, seq, d_model)
        K = X @ self.W_K + self.b_K
        V = X @ self.W_V + self.b_V

        Q = self._split_heads(Q, batch)  # (batch, h, seq, d_k)
        K = self._split_heads(K, batch)
        V = self._split_heads(V, batch)

        # Scaled dot-product attention
        scale  = np.sqrt(self.d_k)
        scores = Q @ K.transpose(0, 1, 3, 2) / scale   # (batch, h, seq, seq)
        attn   = softmax(scores)                         # attention weights

        # Weighted sum of values
        ctx    = attn @ V                                # (batch, h, seq, d_k)
        ctx    = self._merge_heads(ctx, batch)           # (batch, seq, d_model)
        out    = ctx @ self.W_O + self.b_O              # (batch, seq, d_model)

        self._cache = {"X": X, "Q": Q, "K": K, "V": V,
                       "attn": attn, "ctx": ctx, "scores": scores}
        return out, attn


# ── Custom MLP Classifier (from scratch) ──────────────────────────────────────

class CustomMLP:
    """
    Feed-forward classifier with GELU activations.
    Input: pooled transformer output (batch, d_model)
    Output: (batch, n_classes) probabilities
    """

    def __init__(self, input_dim, hidden_dims, output_dim,
                 lr=1e-3, momentum=0.9, lam=1e-4,
                 dropout_rates=None, lr_decay=0.98):

        self.dims    = [input_dim] + hidden_dims + [output_dim]
        self.L       = len(self.dims) - 1
        self.lr      = lr
        self.momentum= momentum
        self.lam     = lam
        self.drops   = dropout_rates or [0.3, 0.2, 0.0]
        self.decay   = lr_decay
        self.history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

        # He initialisation
        self.W = [np.random.randn(self.dims[i], self.dims[i+1]).astype(np.float32)
                  * np.sqrt(2.0 / self.dims[i]) for i in range(self.L)]
        self.b = [np.zeros((1, self.dims[i+1]), dtype=np.float32) for i in range(self.L)]

        # Momentum buffers
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    def _forward(self, X, training=True):
        self._A = [X]; self._Z = []; self._masks = []
        A = X
        for i in range(self.L - 1):
            Z = A @ self.W[i] + self.b[i]
            A = gelu(Z)
            if training and i < len(self.drops) and self.drops[i] > 0:
                p    = self.drops[i]
                mask = (np.random.rand(*A.shape) > p).astype(np.float32) / (1.0 - p)
                A   *= mask
            else:
                mask = np.ones_like(A, dtype=np.float32)
            self._Z.append(Z); self._A.append(A); self._masks.append(mask)

        Z_out = A @ self.W[-1] + self.b[-1]
        A_out = softmax(Z_out)
        self._Z.append(Z_out); self._A.append(A_out)
        self._masks.append(np.ones_like(A_out, dtype=np.float32))
        return A_out

    def _backward(self, Y_oh):
        m  = Y_oh.shape[0]
        dW = [None] * self.L; db = [None] * self.L
        dZ = (self._A[-1] - Y_oh) / m
        for i in reversed(range(self.L)):
            dW[i] = self._A[i].T @ dZ + (self.lam / m) * self.W[i]
            db[i] = dZ.sum(axis=0, keepdims=True)
            if i > 0:
                dA = dZ @ self.W[i].T
                dA *= self._masks[i - 1]
                dZ  = dA * gelu_grad(self._Z[i - 1])
        for i in range(self.L):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * dW[i]
            self.vb[i] = self.momentum * self.vb[i] - self.lr * db[i]
            self.W[i] += self.vW[i]; self.b[i] += self.vb[i]

    def fit(self, X_tr, y_tr, X_val, y_val,
            epochs=30, batch_size=64, patience=6, verbose=True):
        n_cls = self.dims[-1]
        def oh(y):
            o = np.zeros((len(y), n_cls), dtype=np.float32)
            o[np.arange(len(y)), y] = 1.0
            return o
        best_loss = np.inf; best_W = None; best_b = None; stale = 0
        for ep in range(1, epochs + 1):
            idx = np.random.permutation(len(X_tr))
            Xs, ys = X_tr[idx], y_tr[idx]
            for s in range(0, len(Xs), batch_size):
                Xb = Xs[s:s+batch_size]; Yb = oh(ys[s:s+batch_size])
                self._forward(Xb, training=True); self._backward(Yb)
            Yh_tr  = self._forward(X_tr,  training=False)
            Yh_val = self._forward(X_val, training=False)
            def ce(yh, y_oh):
                return -np.sum(y_oh * np.log(yh + 1e-9)) / len(y_oh)
            tl = ce(Yh_tr, oh(y_tr)); vl = ce(Yh_val, oh(y_val))
            ta = (Yh_tr.argmax(1) == y_tr).mean()
            va = (Yh_val.argmax(1) == y_val).mean()
            self.history["train_loss"].append(float(tl))
            self.history["val_loss"].append(float(vl))
            self.history["train_acc"].append(float(ta))
            self.history["val_acc"].append(float(va))
            if verbose:
                print(f"Epoch {ep:>3}/{epochs}  loss={tl:.4f}  val_loss={vl:.4f}  acc={ta:.4f}  val_acc={va:.4f}")
            if vl < best_loss - 1e-5:
                best_loss = vl
                best_W = [w.copy() for w in self.W]
                best_b = [b.copy() for b in self.b]
                stale  = 0
            else:
                stale += 1
                if stale >= patience:
                    if verbose: print(f"  Early stopping at epoch {ep}")
                    break
            self.lr *= self.decay
        if best_W: self.W, self.b = best_W, best_b

    def predict_proba(self, X):
        return self._forward(X, training=False)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# ── Full Hybrid Model ─────────────────────────────────────────────────────────

class HybridEmotionClassifier:
    """
    Full hybrid pipeline:
      DistilBERT → Multi-Head Attention → LayerNorm → Custom MLP
    """

    def __init__(self, n_classes=6, n_heads=4, mlp_dims=None, device="cpu"):
        self.n_classes = n_classes
        self.device    = device
        self.attention = MultiHeadAttention(d_model=768, num_heads=n_heads)
        self.layernorm = LayerNorm(dim=768)
        self.mlp       = CustomMLP(
            input_dim   = 768,
            hidden_dims = mlp_dims or [256, 128, 64],
            output_dim  = n_classes,
            lr          = 1e-3,
            dropout_rates=[0.3, 0.2, 0.0],
        )
        self._bert       = None
        self._tokenizer  = None

    def load_transformer(self):
        """Load DistilBERT — called once at start of training."""
        from transformers import DistilBertTokenizer, DistilBertModel
        import torch
        print("  Loading DistilBERT...")
        self._tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self._bert      = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self._bert.eval()
        self._torch = torch
        print("  DistilBERT loaded.")

    def encode(self, texts, batch_size=32):
        """
        Encode raw texts → mean-pooled DistilBERT embeddings (numpy).
        Shape: (N, 768)
        """
        import torch
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i+batch_size])
            enc   = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            )
            with torch.no_grad():
                out = self._bert(**enc)
            # Mean pooling over token dimension
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            all_embeddings.append(emb.numpy())
            if (i // batch_size) % 5 == 0:
                print(f"    Encoded {min(i+batch_size, len(texts))}/{len(texts)} samples...")
        return np.vstack(all_embeddings).astype(np.float32)

    def apply_attention(self, X):
        """
        Apply custom Multi-Head Attention on top of BERT embeddings.
        X: (N, 768) → expand to (N, 1, 768) → attention → squeeze → LayerNorm
        """
        X_seq = X[:, np.newaxis, :]                 # (N, 1, 768) treat each sample as seq_len=1
        out, attn_weights = self.attention.forward(X_seq)
        out   = out.squeeze(1)                       # (N, 768)
        out   = self.layernorm.forward(out + X)      # residual connection + LayerNorm
        return out, attn_weights

    def fit(self, X_emb, y_tr, X_val_emb, y_val, **kwargs):
        """Train the custom MLP on top of precomputed embeddings."""
        X_att,   _ = self.apply_attention(X_emb)
        X_val_att, _ = self.apply_attention(X_val_emb)
        self.mlp.fit(X_att, y_tr, X_val_att, y_val, **kwargs)

    def predict_proba(self, X_emb):
        X_att, _ = self.apply_attention(X_emb)
        return self.mlp.predict_proba(X_att)

    def predict(self, X_emb):
        return self.predict_proba(X_emb).argmax(axis=1)
