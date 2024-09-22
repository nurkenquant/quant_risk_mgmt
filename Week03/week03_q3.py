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
def simulate_from_covariance(cov_matrix, num_simulations=1000, num_assets=100):
    mean = np.zeros(num_assets)  # Assuming zero mean for daily returns
    simulated_data = np.random.multivariate_normal(mean, cov_matrix, num_simulations)
    return simulated_data

# Simulate 1000 data points based on the covariance matrix
simulated_data_cov = simulate_from_covariance(cov_matrix, num_simulations=1000, num_assets=cov_matrix.shape[0])

# Displaying the shape and first few rows of the simulated data with labels
num_simulations, num_assets = simulated_data_cov.shape
print(f"Simulated Data Shape: {num_simulations} simulations, {num_assets} assets")
print("\nThe Simulated Data (Covariance-based):")
print(pd.DataFrame(simulated_data_cov, columns=daily_returns.columns))

# Function to simulate using PCA
def simulate_using_pca(cov_matrix, num_simulations=1000, variance_explained=0.95):
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

# Simulate 1000 data points using PCA with 95% variance explained
simulated_data_pca, num_components_used = simulate_using_pca(cov_matrix, num_simulations=1000, variance_explained=0.95)

# Display the number of PCA components used and the shape of the simulated data
num_simulations, num_assets = simulated_data_pca.shape
print(f"Number of PCA Components Used: {num_components_used}")
print(f"Simulated Data Shape: {num_simulations} simulations, {num_assets} assets\n")

# Displaying the first few rows of the PCA-based simulated data for clarity
print("The Simulated Data (PCA-based):")
print(pd.DataFrame(simulated_data_pca, columns=daily_returns.columns))


# Function to simulate data using a package (scipy) for comparison
def simulate_with_scipy(cov_matrix, num_simulations=1000):
    mean = np.zeros(cov_matrix.shape[0])
    scipy_simulated_data = multivariate_normal.rvs(mean=mean, cov=cov_matrix, size=num_simulations)
    return scipy_simulated_data

# Simulate using scipy
scipy_simulated_data = simulate_with_scipy(cov_matrix, num_simulations=1000)

# Calculate and compare the means and variances of the first asset between custom and scipy simulations
comparison_df = pd.DataFrame({
    "Custom Simulation (Covariance)": simulated_data_cov[:, 0],
    "Custom Simulation (PCA)": simulated_data_pca[:, 0],
    "Scipy Simulation": scipy_simulated_data[:, 0]
})

# Calculate and display comparison statistics
comparison_statistics = comparison_df.describe()
print("\nComparison Statistics:")
print(comparison_statistics)


#Generating Correlation Matrices and Variance Vectors

# 1. Standard Pearson Correlation and Variance
pearson_correlation_matrix = daily_returns.corr()
standard_variance_vector = daily_returns.var()

# Function to calculate exponentially weighted variance
def exponentially_weighted_variance(returns, lambda_value=0.97):
    ew_variance_vector = returns.ewm(span=(1/(1-lambda_value))).var().iloc[-1]
    return ew_variance_vector

# Function to calculate exponentially weighted correlation
def exponentially_weighted_correlation(returns, lambda_value=0.97):
    returns_mean = returns.ewm(span=(1/(1-lambda_value))).mean()
    demeaned = returns - returns_mean
    ew_cov_matrix = demeaned.ewm(span=(1/(1-lambda_value))).cov(pairwise=True)
    ew_correlation_matrix = ew_cov_matrix.iloc[-1]
    return ew_correlation_matrix

# 2. Exponentially Weighted Variance and Correlation
ew_variance_vector = exponentially_weighted_variance(daily_returns, lambda_value=0.97)
ew_correlation_matrix = exponentially_weighted_correlation(daily_returns, lambda_value=0.97)

# Correcting the exponentially weighted correlation matrix to proper format
ew_correlation_matrix_corrected = pd.DataFrame(index=daily_returns.columns, columns=daily_returns.columns)
for index, value in ew_correlation_matrix.items():
    if isinstance(index, tuple):
        ew_correlation_matrix_corrected.loc[index[0], index[1]] = value

# Fill diagonal with 1.0 as correlation of an asset with itself is 1
np.fill_diagonal(ew_correlation_matrix_corrected.values, 1.0)

# Convert matrix to numeric to handle NaNs or missing values
ew_correlation_matrix_corrected = ew_correlation_matrix_corrected.apply(pd.to_numeric, errors='coerce')
ew_correlation_matrix_corrected = ew_correlation_matrix_corrected.fillna(0)

# Function to create covariance matrix from correlation matrix and variance vector
def create_covariance_matrix(correlation_matrix, variance_vector):
    # Convert variance vector to standard deviation
    std_dev_vector = np.sqrt(variance_vector)
    
    # Outer product of standard deviations to form the variance-covariance matrix
    variance_covariance_matrix = np.outer(std_dev_vector, std_dev_vector)
    
    # Multiply element-wise with the correlation matrix
    covariance_matrix = correlation_matrix * variance_covariance_matrix
    return covariance_matrix

# Creating the four covariance matrices
cov_matrix_1 = create_covariance_matrix(pearson_correlation_matrix, standard_variance_vector)
cov_matrix_2 = create_covariance_matrix(pearson_correlation_matrix, ew_variance_vector)
cov_matrix_3 = create_covariance_matrix(ew_correlation_matrix_corrected, standard_variance_vector)
cov_matrix_4 = create_covariance_matrix(ew_correlation_matrix_corrected, ew_variance_vector)

# Display the first few rows of each covariance matrix for verification
print("\nCovariance Matrix 1 (Pearson Correlation + Standard Variance):")
print(pd.DataFrame(cov_matrix_1, index=daily_returns.columns, columns=daily_returns.columns).head())

print("\nCovariance Matrix 2 (Pearson Correlation + EW Variance):")
print(pd.DataFrame(cov_matrix_2, index=daily_returns.columns, columns=daily_returns.columns).head())

print("\nCovariance Matrix 3 (EW Correlation + Standard Variance):")
print(pd.DataFrame(cov_matrix_3, index=daily_returns.columns, columns=daily_returns.columns).head())

print("\nCovariance Matrix 4 (EW Correlation + EW Variance):")
print(pd.DataFrame(cov_matrix_4, index=daily_returns.columns, columns=daily_returns.columns).head())