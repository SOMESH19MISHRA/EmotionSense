import numpy as np
import pytest

def test_numpy_available():
    assert np.__version__ is not None

def test_basic_math():
    x = np.array([1.0, 2.0, 3.0])
    assert x.sum() == 6.0

def test_softmax():
    from scipy.special import softmax
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    assert abs(result.sum() - 1.0) < 1e-6

def test_sklearn_available():
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(['joy', 'sadness', 'anger'])
    assert len(le.classes_) == 3
