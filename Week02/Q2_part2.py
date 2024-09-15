import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy import stats

# Load the dataset
file_path = 'problem2.csv'
data = pd.read_csv(file_path)

# Define the negative log-likelihood function assuming normality
def negative_log_likelihood_normal(params, X, y):
    beta_0, beta_1, sigma = params
    y_pred = beta_0 + beta_1 * X
    n = len(y)
    residuals = y - y_pred
    log_likelihood = -n/2 * np.log(2 * np.pi * sigma**2) - np.sum(residuals**2) / (2 * sigma**2)
    return -log_likelihood  # We minimize the negative log-likelihood

# Define the negative log-likelihood function assuming t-distribution
def negative_log_likelihood_t(params, X, y, df=3):  # Using df=3 for the t-distribution
    beta_0, beta_1, sigma = params
    y_pred = beta_0 + beta_1 * X
    residuals = y - y_pred
    log_likelihood = np.sum(stats.t.logpdf(residuals / sigma, df=df) - np.log(sigma))
    return -log_likelihood  # We minimize the negative log-likelihood

# Initial guesses for MLE parameters: [beta_0, beta_1, sigma]
initial_guess = [0, 0, 1]

# Perform the optimization to minimize the negative log-likelihood (normal distribution)
mle_result_normal = minimize(negative_log_likelihood_normal, initial_guess, args=(data['x'], data['y']), bounds=[(None, None), (None, None), (0.0001, None)])

# Perform the optimization to minimize the negative log-likelihood (t-distribution)
mle_result_t = minimize(negative_log_likelihood_t, initial_guess, args=(data['x'], data['y']), bounds=[(None, None), (None, None), (0.0001, None)])

# Extract the MLE parameters (beta_0, beta_1, sigma) for both normal and t-distribution
beta_mle_normal_0, beta_mle_normal_1, sigma_mle_normal = mle_result_normal.x
beta_mle_t_0, beta_mle_t_1, sigma_mle_t = mle_result_t.x

# Display results for comparison
results = f"""
Model Comparison Results:
MLE Normal Intercept (Beta 0): {beta_mle_normal_0:.4f}
MLE Normal Slope (Beta 1): {beta_mle_normal_1:.4f}
MLE Normal Sigma: {sigma_mle_normal:.4f}
MLE T-distribution Intercept (Beta 0): {beta_mle_t_0:.4f}
MLE T-distribution Slope (Beta 1): {beta_mle_t_1:.4f}
MLE T-distribution Sigma: {sigma_mle_t:.4f}
"""

print(results)

# Function to calculate AIC
def calculate_aic(log_likelihood, num_params, n):
    return 2 * num_params - 2 * log_likelihood

# Calculate log-likelihood for the normal distribution model
def log_likelihood_normal(params, X, y):
    beta_0, beta_1, sigma = params
    y_pred = beta_0 + beta_1 * X
    residuals = y - y_pred
    n = len(y)
    log_likelihood = -n / 2 * np.log(2 * np.pi * sigma**2) - np.sum(residuals**2) / (2 * sigma**2)
    return log_likelihood

# Calculate log-likelihood for the t-distribution model
def log_likelihood_t(params, X, y, df=3):
    beta_0, beta_1, sigma = params
    y_pred = beta_0 + beta_1 * X
    residuals = y - y_pred
    log_likelihood = np.sum(stats.t.logpdf(residuals / sigma, df=df) - np.log(sigma))
    return log_likelihood

# Log-likelihood values
log_likelihood_normal_val = log_likelihood_normal(mle_result_normal.x, data['x'], data['y'])
log_likelihood_t_val = log_likelihood_t(mle_result_t.x, data['x'], data['y'])

# Number of parameters (beta_0, beta_1, and sigma)
num_params = 3
n = len(data['y'])

# Calculate AIC for both models
aic_normal = calculate_aic(log_likelihood_normal_val, num_params, n)
aic_t = calculate_aic(log_likelihood_t_val, num_params, n)

# Function to calculate BIC
def calculate_bic(log_likelihood, num_params, n):
    return num_params * np.log(n) - 2 * log_likelihood

# Calculate BIC for both models
bic_normal = calculate_bic(log_likelihood_normal_val, num_params, n)
bic_t = calculate_bic(log_likelihood_t_val, num_params, n)


# Display results of Log-Likelihood, AIC/BIC
comparison_results = f"""
Log-Likelihood, AIC, And BIC Comparison:
   Normal Distribution Model:
       Log-Likelihood: {log_likelihood_normal_val:.4f}
       AIC: {aic_normal:.4f}
       BIC: {bic_normal:.4f}
   
   T-Distribution Model:
      Log-Likelihood: {log_likelihood_t_val:.4f}
      AIC: {aic_t:.4f}
      BIC: {bic_t:.4f}
"""

print(comparison_results)
