import numpy as np
import scipy.linalg
import time

def chol_psd(matrix):
    """Performs the Cholesky decomposition on a matrix assumed to be PSD."""
    n = matrix.shape[0]
    root = np.zeros_like(matrix)
    
    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])
        
        temp = matrix[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        
        root[j, j] = np.sqrt(temp)
        
        if root[j, j] != 0.0:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (matrix[i, j] - s) * ir
    
    return root

def near_psd(matrix, epsilon=0.0):
    """Corrects a matrix to be positive semi-definite."""
    n = matrix.shape[0]
    inv_sd = None
    out = matrix.copy()

    # Convert to correlation matrix if covariance is provided
    if not np.allclose(np.diag(out), 1.0):
        inv_sd = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = inv_sd @ out @ inv_sd

    # Eigen decomposition and modify eigenvalues
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / (vecs ** 2 @ vals)
    T = np.diag(np.sqrt(T))
    L = np.diag(np.sqrt(vals))
    B = T @ vecs @ L
    out = B @ B.T

    # Revert to original scale if covariance was provided
    if inv_sd is not None:
        inv_sd = np.diag(1.0 / np.diag(inv_sd))
        out = inv_sd @ out @ inv_sd

    return out

#Generate a non-PSD correlation matrix
n = 500
sigma = np.full((n, n), 0.9)
np.fill_diagonal(sigma, 1.0)
sigma[0, 1] = 0.7357
sigma[1, 0] = 0.7357

# Check the matrix to confirm its non-PSD nature by inspecting eigenvalues
initial_eigenvalues = np.linalg.eigvals(sigma)
non_psd = any(initial_eigenvalue <= 0 for initial_eigenvalue in initial_eigenvalues)

# Print statements to illustrate the results
print("Step 1: Generating a non-PSD matrix of size 500x500.")
print("Initial eigenvalues of the matrix show that it is non-PSD (some values are negative or zero):")
print(initial_eigenvalues)
print(f"Is the matrix non-PSD? {'Yes, it has non-positive eigenvalues.' if non_psd else 'No, it is already PSD.'}\n")

# Correct the matrix using near_psd method
print("Step 2: Correcting the matrix using the 'near_psd' method...")
near_psd_matrix = near_psd(sigma)
near_psd_eigenvalues = np.linalg.eigvals(near_psd_matrix)
near_psd_is_psd = all(near_psd_eigenvalue > 0 for near_psd_eigenvalue in near_psd_eigenvalues)
print("Corrected matrix eigenvalues after applying 'near_psd' method:")
print(near_psd_eigenvalues)
print(f"Is the matrix PSD after correction? {'Yes, it is now PSD.' if near_psd_is_psd else 'No, the correction failed.'}\n")

def higham_nearest_psd(matrix, max_iterations=100, tol=1e-6):
    """Finds the nearest positive semi-definite matrix using Higham's method."""
    n = matrix.shape[0]
    Y = matrix.copy()
    delta_S = np.zeros_like(matrix)
    gamma = np.inf

    for _ in range(max_iterations):
        R = Y - delta_S
        X = (R + R.T) / 2
        eigvals, eigvecs = np.linalg.eigh(X)
        eigvals[eigvals < 0] = 0
        Y = eigvecs @ np.diag(eigvals) @ eigvecs.T

        delta_S = Y - X
        gamma_new = np.linalg.norm(Y - matrix, 'fro')
        
        if np.abs(gamma - gamma_new) < tol:
            break
        
        gamma = gamma_new

    return Y

# Correct the matrix using Higham's method
print("Step 3: Correcting the matrix using Higham's method...")

# Apply Higham's method to the non-PSD matrix
higham_psd_matrix = higham_nearest_psd(sigma)

# Check if the resulting matrix is PSD
higham_psd_eigenvalues = np.linalg.eigvals(higham_psd_matrix)
higham_psd_is_psd = all(higham_psd_eigenvalue > 0 for higham_psd_eigenvalue in higham_psd_eigenvalues)
print("Corrected matrix eigenvalues after applying Higham's method:")
print(higham_psd_eigenvalues)
print(f"Is the matrix PSD after correction? {'Yes, it is now PSD.' if higham_psd_is_psd else 'No, the correction failed.'}\n")

# Final comparison of Frobenius norms and runtimes
print("Step 4: Comparing the Frobenius norms and runtime for both methods.")
frobenius_near_psd = np.linalg.norm(sigma - near_psd_matrix, 'fro')
frobenius_higham = np.linalg.norm(sigma - higham_psd_matrix, 'fro')
start_time_near_psd = time.time()
near_psd(sigma)
time_near_psd = time.time() - start_time_near_psd
start_time_higham = time.time()
higham_nearest_psd(sigma)
time_higham = time.time() - start_time_higham

print(f"Frobenius Norm (Near PSD): {frobenius_near_psd}")
print(f"Frobenius Norm (Higham): {frobenius_higham}")
print(f"Runtime (Near PSD Method): {time_near_psd:.4f} seconds")
print(f"Runtime (Higham's Method): {time_higham:.4f} seconds")

# Final results comparison dictionary for clarity
comparison_results = {
    "Frobenius Norm - Near PSD": frobenius_near_psd,
    "Frobenius Norm - Higham": frobenius_higham,
    "Runtime - Near PSD (s)": time_near_psd,
    "Runtime - Higham (s)": time_higham
}

