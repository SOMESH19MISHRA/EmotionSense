"""
neural_network.py
-----------------
Multi-Layer Perceptron built entirely from scratch using NumPy only.
No scikit-learn, no PyTorch, no TensorFlow — pure math.

Architecture:
  Input → Dense(256, ReLU) → Dropout(0.3)
        → Dense(128, ReLU) → Dropout(0.2)
        → Dense(64,  ReLU)
        → Dense(N_classes, Softmax)

Features implemented from scratch:
  - He weight initialisation
  - ReLU + numerically-stable Softmax
  - Inverted Dropout regularisation
  - Cross-entropy loss with L2 weight penalty
  - Mini-batch SGD with Momentum
  - Per-epoch learning rate decay
  - Early stopping with best-weight restoration
"""

import numpy as np


# ── Activations ───────────────────────────────────────────────────────────────

def relu(Z):
    return np.maximum(0.0, Z)

def relu_grad(dA, Z):
    """Gradient of ReLU: pass through where Z > 0, zero elsewhere."""
    dZ = dA.copy()
    dZ[Z <= 0] = 0.0
    return dZ

def softmax(Z):
    """Numerically stable row-wise softmax."""
    Z_s = Z - Z.max(axis=1, keepdims=True)   # subtract max for stability
    e   = np.exp(Z_s)
    return e / e.sum(axis=1, keepdims=True)


# ── Loss ──────────────────────────────────────────────────────────────────────

def cross_entropy(Y_hat, Y_oh, weights, lam):
    """Cross-entropy loss + L2 regularisation."""
    m   = Y_oh.shape[0]
    ce  = -np.sum(Y_oh * np.log(Y_hat + 1e-9)) / m
    l2  = (lam / (2 * m)) * sum(np.sum(W ** 2) for W in weights)
    return ce + l2


# ── Model ─────────────────────────────────────────────────────────────────────

class EmotionMLP:
    """
    Custom fully-connected neural network for multi-class emotion detection.

    Parameters
    ----------
    input_dim     : number of TF-IDF input features
    hidden_dims   : list of hidden-layer widths  e.g. [256, 128, 64]
    output_dim    : number of emotion classes
    lr            : initial learning rate
    momentum      : SGD momentum coefficient (β)
    lam           : L2 regularisation strength (λ)
    dropout_rates : dropout probability per hidden layer
    lr_decay      : multiply lr by this value after each epoch
    """

    def __init__(self, input_dim, hidden_dims, output_dim,
                 lr=0.05, momentum=0.9, lam=1e-4,
                 dropout_rates=None, lr_decay=0.97):

        self.dims    = [input_dim] + hidden_dims + [output_dim]
        self.L       = len(self.dims) - 1          # number of weight layers
        self.lr      = lr
        self.momentum= momentum
        self.lam     = lam
        self.drops   = (dropout_rates or [0.3, 0.2, 0.0])
        self.decay   = lr_decay
        self.history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

        # ── He initialisation ─────────────────────────────────────────────
        self.W = [np.random.randn(self.dims[i], self.dims[i+1]) * np.sqrt(2.0 / self.dims[i])
                  for i in range(self.L)]
        self.b = [np.zeros((1, self.dims[i+1])) for i in range(self.L)]

        # ── Momentum velocity buffers ─────────────────────────────────────
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _forward(self, X, training=True):
        """Run forward pass; cache activations for backprop."""
        self._A, self._Z, self._masks = [X], [], []
        A = X

        for i in range(self.L - 1):            # hidden layers
            Z = A @ self.W[i] + self.b[i]
            A = relu(Z)
            # inverted dropout
            if training and i < len(self.drops) and self.drops[i] > 0:
                p    = self.drops[i]
                mask = (np.random.rand(*A.shape) > p).astype(np.float32) / (1.0 - p)
                A   *= mask
            else:
                mask = np.ones_like(A, dtype=np.float32)
            self._Z.append(Z)
            self._A.append(A)
            self._masks.append(mask)

        # output layer — softmax, no dropout
        Z_out = A @ self.W[-1] + self.b[-1]
        A_out = softmax(Z_out)
        self._Z.append(Z_out)
        self._A.append(A_out)
        self._masks.append(np.ones_like(A_out, dtype=np.float32))
        return A_out

    # ── Backward pass ─────────────────────────────────────────────────────────

    def _backward(self, Y_oh):
        """Backprop and update weights with SGD + momentum."""
        m  = Y_oh.shape[0]
        dW = [None] * self.L
        db = [None] * self.L

        # combined softmax + cross-entropy gradient: (Ŷ - Y) / m
        dZ = (self._A[-1] - Y_oh) / m

        for i in reversed(range(self.L)):
            A_prev = self._A[i]
            dW[i]  = A_prev.T @ dZ + (self.lam / m) * self.W[i]
            db[i]  = dZ.sum(axis=0, keepdims=True)

            if i > 0:
                dA = dZ @ self.W[i].T
                dA *= self._masks[i - 1]          # apply dropout mask
                dZ  = relu_grad(dA, self._Z[i - 1])

        # SGD + momentum update: v = β·v - α·∇W;  W += v
        for i in range(self.L):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * dW[i]
            self.vb[i] = self.momentum * self.vb[i] - self.lr * db[i]
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]

    # ── Training loop ─────────────────────────────────────────────────────────

    def fit(self, X_tr, y_tr, X_val, y_val,
            epochs=50, batch_size=256, patience=7, verbose=True):

        n_cls  = self.dims[-1]

        def one_hot(y):
            oh = np.zeros((len(y), n_cls), dtype=np.float32)
            oh[np.arange(len(y)), y] = 1.0
            return oh

        Y_val_oh   = one_hot(y_val)
        best_loss  = np.inf
        best_W, best_b = None, None
        stale      = 0

        for ep in range(1, epochs + 1):
            # shuffle
            idx  = np.random.permutation(len(X_tr))
            Xs, ys = X_tr[idx], y_tr[idx]

            # mini-batch gradient descent
            for s in range(0, len(Xs), batch_size):
                Xb = Xs[s: s + batch_size]
                Yb = one_hot(ys[s: s + batch_size])
                self._forward(Xb, training=True)
                self._backward(Yb)

            # ── Metrics ───────────────────────────────────────────────────
            Yh_tr  = self._forward(X_tr,  training=False)
            Yh_val = self._forward(X_val, training=False)
            tr_loss  = cross_entropy(Yh_tr,  one_hot(y_tr), self.W, self.lam)
            vl_loss  = cross_entropy(Yh_val, Y_val_oh,      self.W, self.lam)
            tr_acc   = (Yh_tr.argmax(1)  == y_tr ).mean()
            vl_acc   = (Yh_val.argmax(1) == y_val).mean()

            self.history["train_loss"].append(float(tr_loss))
            self.history["val_loss"].append(float(vl_loss))
            self.history["train_acc"].append(float(tr_acc))
            self.history["val_acc"].append(float(vl_acc))

            if verbose:
                print(f"Epoch {ep:>3}/{epochs}  "
                      f"loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
                      f"acc={tr_acc:.4f}  val_acc={vl_acc:.4f}")

            # ── Early stopping ────────────────────────────────────────────
            if vl_loss < best_loss - 1e-5:
                best_loss   = vl_loss
                best_W      = [w.copy() for w in self.W]
                best_b      = [b.copy() for b in self.b]
                stale       = 0
            else:
                stale += 1
                if stale >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {ep} — restoring best weights.")
                    break

            # learning rate decay
            self.lr *= self.decay

        if best_W:
            self.W, self.b = best_W, best_b

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X):
        return self._forward(X, training=False)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
