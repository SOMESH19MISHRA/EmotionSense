"""Unit tests for EmotionSense preprocessing and neural network."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.preprocess import clean_text, TextVectorizer
from model.neural_network import EmotionMLP

def test_clean_removes_urls():
    assert "http" not in clean_text("visit http://example.com now")

def test_clean_expands_contractions():
    assert "cannot" in clean_text("I can't do this")

def test_clean_lowercases():
    assert clean_text("HELLO WORLD") == clean_text("hello world")

def test_vectorizer_shape():
    vec = TextVectorizer(max_features=50)
    X = vec.fit_transform(["i am happy today", "this is terrible", "great day", "so sad"])
    assert X.shape[0] == 4 and X.shape[1] <= 50

def test_mlp_softmax_sums_to_one():
    np.random.seed(0)
    X = np.random.rand(8, 40).astype(np.float32)
    m = EmotionMLP(input_dim=40, hidden_dims=[32, 16], output_dim=6)
    p = m.predict_proba(X)
    assert p.shape == (8, 6)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)

def test_mlp_predict_valid_class():
    np.random.seed(0)
    X = np.random.rand(5, 40).astype(np.float32)
    m = EmotionMLP(input_dim=40, hidden_dims=[32], output_dim=6)
    assert all(0 <= p < 6 for p in m.predict(X))
