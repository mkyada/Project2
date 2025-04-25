import numpy as np
import pytest
from GradientBoost.model.GradientBoostClassifier import GradientBoostClassifier
from .test_cases import load_small_binary, load_constant_labels

def test_simple_separable():
    X, y = load_small_binary()
    clf = GradientBoostClassifier(n_estimators=10, learning_rate=0.5, max_depth=1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert np.array_equal(y_pred, y)

def test_all_same_label():
    X, y = load_constant_labels()
    clf = GradientBoostClassifier(n_estimators=5, learning_rate=0.1, max_depth=2)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert np.all(y_pred == 0)

def test_predict_proba_shape():
    X, y = load_small_binary()
    clf = GradientBoostClassifier(n_estimators=5)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

def test_zero_estimators_uses_initial_guess():
    X, y = load_small_binary()
    clf = GradientBoostClassifier(n_estimators=0, learning_rate=0.1)
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    p0 = np.clip(np.mean(y), 1e-5, 1 - 1e-5)
    assert np.allclose(proba, p0, atol=1e-8)

def test_single_sample():
    X = np.array([[42., -1.]])
    y = np.array([1])
    clf = GradientBoostClassifier(n_estimators=5, learning_rate=0.1, max_depth=1)
    clf.fit(X, y)
    # With only one sample of class “1”, prediction should be 1
    assert clf.predict(X)[0] == 1

def test_one_dimensional_data():
    X = np.array([[0.], [1.], [2.], [3.]])
    y = np.array([0, 0, 1, 1])
    clf = GradientBoostClassifier(n_estimators=20, learning_rate=0.2, max_depth=1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    # Perfectly separable in 1D
    assert np.array_equal(y_pred, y)

def test_non_linear_boundary():
    # XOR‐style + two extra zeros
    X = np.array([[0,0],[0,1],[1,0],[1,1],[2,2],[-1,-1]], dtype=float)
    y = np.array([0,1,1,0,0,0], dtype=int)
    # Increase capacity to learn the pattern
    clf = GradientBoostClassifier(n_estimators=100, learning_rate=0.1, max_depth=2)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    # Require at least 4 out of 6 correct (≈67% accuracy)
    assert (y_pred == y).sum() >= 4

@pytest.mark.parametrize("n,d", [(5,2), (10,3), (1,1)])
def test_predict_proba_range(n, d):
    rng = np.random.RandomState(0)
    X = rng.randn(n, d)
    y = rng.randint(0, 2, size=n)
    clf = GradientBoostClassifier(n_estimators=5)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    # Probabilities in [0,1] and rows sum to 1
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
