import numpy as np

def load_small_binary():
    # simple XOR-like toy dataset
    X = np.array([[0,0], [1,1], [0,1], [1,0]], dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)
    return X, y

def load_constant_labels(n=10):
    X = np.random.randn(n, 2)
    y = np.zeros(n, dtype=int)
    return X, y
