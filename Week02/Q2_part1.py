import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy import stats

# Load the dataset
file_path = 'problem2.csv'
data = pd.read_csv(file_path)

# OLS Regression
X_ols = sm.add_constant(data['x'])  # Add a constant term (intercept)
y_ols = data['y']

# Fit the OLS model
ols_model = sm.OLS(y_ols, X_ols).fit()

# Get OLS beta values and standard deviation of residuals (errors)
beta_ols = ols_model.params
residuals_ols = ols_model.resid
std_ols = np.std(residuals_ols)

# Format and display the beta values and residuals standard deviation
OLS_output = f"""
OLS Results:
Intercept (Beta 0): {beta_ols.iloc[0]:.4f}
Slope (Beta 1): {beta_ols.iloc[1]:.4f}
Standard Deviation of Residuals (OLS): {std_ols:.4f}
"""
print(OLS_output)


# Define the negative log-likelihood function assuming normality
def negative_log_likelihood(params, X, y):
    beta_0, beta_1, sigma = params
    y_pred = beta_0 + beta_1 * X
    n = len(y)
    residuals = y - y_pred
    log_likelihood = -n/2 * np.log(2 * np.pi * sigma**2) - np.sum(residuals**2) / (2 * sigma**2)
    return -log_likelihood  # We minimize the negative log-likelihood

# Initial guesses for MLE parameters: [beta_0, beta_1, sigma]
initial_guess = [0, 0, 1]

# Perform the optimization to minimize the negative log-likelihood
mle_result = minimize(negative_log_likelihood, initial_guess, args=(data['x'], data['y']), bounds=[(None, None), (None, None), (0.0001, None)])

# Extract the MLE parameters (beta_0, beta_1, sigma)
beta_mle_0, beta_mle_1, sigma_mle = mle_result.x

# Format and display the MLE beta values and fitted sigma
MLE_output = f"""
MLE Results:
Intercept (Beta 0): {beta_mle_0:.4f}
Slope (Beta 1): {beta_mle_1:.4f}
MLE Estimate of 𝜎: {sigma_mle:.4f}
"""
print(MLE_output)
