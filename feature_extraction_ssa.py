import numpy as np
from scipy.linalg import svd

def ssa_decompose(signal, window_length):
    N = len(signal)
    K = N - window_length + 1
    X = np.column_stack([signal[i:i+K] for i in range(window_length)])
    U, s, Vh = svd(X)
    return U, s, Vh

def reconstruct_signal(U, s, Vh, components):
    X_reconstructed = sum(s[i] * np.outer(U[:, i], Vh[i, :]) for i in components)
    return X_reconstructed.mean(axis=1)