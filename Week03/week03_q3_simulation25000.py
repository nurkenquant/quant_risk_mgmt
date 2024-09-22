import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import time

# Load the DailyReturn.csv file
file_path = "DailyReturn.csv"
daily_returns = pd.read_csv(file_path)

# Calculate the covariance matrix of the daily returns
cov_matrix = daily_returns.cov()

# Function to simulate from a covariance matrix
def simulate_from_covariance(cov_matrix, num_simulations=25000, num_assets=100):
    mean = np.zeros(num_assets)  # Assuming zero mean for daily returns
    simulated_data = np.random.multivariate_normal(mean, cov_matrix, num_simulations)
    return simulated_data

# Function to simulate using PCA
def simulate_using_pca(cov_matrix, num_simulations=25000, variance_explained=1.0):
    # Calculate PCA
    pca = PCA()
    pca.fit(cov_matrix)
    
    # Determine the number of components to use based on the explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.searchsorted(cumulative_variance, variance_explained) + 1
    
    # Transform covariance matrix using selected components
    transformed_cov_matrix = pca.components_[:num_components].T @ np.diag(pca.explained_variance_[:num_components]) @ pca.components_[:num_components]
    
    # Simulate data using reduced covariance matrix
    mean = np.zeros(cov_matrix.shape[0])
    simulated_data_pca = np.random.multivariate_normal(mean, transformed_cov_matrix, num_simulations)
    
    return simulated_data_pca, num_components

# Function to calculate Frobenius Norm
def frobenius_norm(matrix1, matrix2):
    # Convert matrices to numpy arrays to avoid deprecation warnings
    if isinstance(matrix1, pd.DataFrame):
        matrix1 = matrix1.values
    if isinstance(matrix2, pd.DataFrame):
        matrix2 = matrix2.values
    return np.sqrt(np.sum((matrix1 - matrix2) ** 2))

# Function to measure runtime and accuracy
def simulate_and_compare(cov_matrix, description, variance_explained=1.0):
    start_time = time.time()
    if variance_explained == 1.0:
        # Direct Simulation
        simulated_data = simulate_from_covariance(cov_matrix, num_simulations=25000, num_assets=cov_matrix.shape[0])
    else:
        # PCA Simulation
        simulated_data, num_components = simulate_using_pca(cov_matrix, num_simulations=25000, variance_explained=variance_explained)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Calculate covariance of simulated data
    simulated_cov_matrix = np.cov(simulated_data, rowvar=False)
    
    # Calculate Frobenius Norm
    frob_norm = frobenius_norm(cov_matrix, simulated_cov_matrix)
    
    return runtime, frob_norm

# Function to create covariance matrix from correlation matrix and variance vector
def create_covariance_matrix(correlation_matrix, variance_vector):
    # Convert variance vector to standard deviation
    std_dev_vector = np.sqrt(variance_vector)
    
    # Outer product of standard deviations to form the variance-covariance matrix
    variance_covariance_matrix = np.outer(std_dev_vector, std_dev_vector)
    
    # Multiply element-wise with the correlation matrix
    covariance_matrix = correlation_matrix * variance_covariance_matrix
    return covariance_matrix

# Covariance Matrices from Step 2
# Standard Pearson Correlation and Variance
pearson_correlation_matrix = daily_returns.corr()
standard_variance_vector = daily_returns.var()
ew_variance_vector = daily_returns.ewm(span=(1/(1-0.97))).var().iloc[-1]
returns_mean = daily_returns.ewm(span=(1/(1-0.97))).mean()
demeaned = daily_returns - returns_mean
ew_cov_matrix = demeaned.ewm(span=(1/(1-0.97))).cov(pairwise=True)
ew_correlation_matrix = ew_cov_matrix.iloc[-1]

ew_correlation_matrix_corrected = pd.DataFrame(index=daily_returns.columns, columns=daily_returns.columns)
for index, value in ew_correlation_matrix.items():
    if isinstance(index, tuple):
        ew_correlation_matrix_corrected.loc[index[0], index[1]] = value

np.fill_diagonal(ew_correlation_matrix_corrected.values, 1.0)
ew_correlation_matrix_corrected = ew_correlation_matrix_corrected.apply(pd.to_numeric, errors='coerce')
ew_correlation_matrix_corrected = ew_correlation_matrix_corrected.fillna(0)

# Create the four covariance matrices
cov_matrix_1 = create_covariance_matrix(pearson_correlation_matrix, standard_variance_vector)
cov_matrix_2 = create_covariance_matrix(pearson_correlation_matrix, ew_variance_vector)
cov_matrix_3 = create_covariance_matrix(ew_correlation_matrix_corrected, standard_variance_vector)
cov_matrix_4 = create_covariance_matrix(ew_correlation_matrix_corrected, ew_variance_vector)

# Collect results
results = []

# Simulate and compare for each covariance matrix
for i, (cov_matrix, desc) in enumerate([
    (cov_matrix_1, "Covariance Matrix 1 (Pearson Correlation + Standard Variance)"),
    (cov_matrix_2, "Covariance Matrix 2 (Pearson Correlation + EW Variance)"),
    (cov_matrix_3, "Covariance Matrix 3 (EW Correlation + Standard Variance)"),
    (cov_matrix_4, "Covariance Matrix 4 (EW Correlation + EW Variance)")
]):
    print(f"Simulating from {desc}:")
    results.append((desc, "Direct", *simulate_and_compare(cov_matrix, "Direct Simulation", variance_explained=1.0)))
    results.append((desc, "PCA 100%", *simulate_and_compare(cov_matrix, "PCA Simulation", variance_explained=1.0)))
    results.append((desc, "PCA 75%", *simulate_and_compare(cov_matrix, "PCA Simulation", variance_explained=0.75)))
    results.append((desc, "PCA 50%", *simulate_and_compare(cov_matrix, "PCA Simulation", variance_explained=0.50)))

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results, columns=["Covariance Matrix", "Simulation Method", "Runtime (seconds)", "Frobenius Norm"])
print(results_df)
