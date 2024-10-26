# Non-PSD matrix adjustments
import numpy as np

def near_psd(matrix, epsilon=1e-8):
    eigval, eigvec = np.linalg.eigh(matrix)
    eigval[eigval < epsilon] = epsilon
    return eigvec @ np.diag(eigval) @ eigvec.T
