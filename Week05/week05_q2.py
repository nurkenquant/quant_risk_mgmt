import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy.optimize import minimize

# Load the data from 'problem1.csv'
data_path = "problem1.csv"
data = pd.read_csv(data_path)

# Extract the returns data
returns = data['x'].values

# parameters
lambda_value = 0.97
confidence_level = 0.99

# Calculate the weights
def calculate_weights(m, lambda_value):
    weights = np.array([(1 - lambda_value) * (lambda_value ** (m - i - 1)) for i in range(m)])
    return weights / weights.sum()

# Calculate the exponentially weighted variance
def ew_variance(data, lambda_value):
    m = len(data)
    weights = calculate_weights(m, lambda_value)
    mean_adjusted_data = np.sqrt(weights) * (data - np.dot(weights, data))
    ew_var = np.dot(mean_adjusted_data, mean_adjusted_data)
    return ew_var

# Define the function to calculate VaR and ES using a normal distribution with an exponentially weighted variance (lambda=0.97)
def calculate_var_es_EW(data, lambda_value, confidence_level):
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES) using a normal distribution
    with exponentially weighted variance.

    Parameters:
        data (array-like): The data series of returns.
        lambda_value (float): The lambda value for exponential weighting.
        confidence_level (float): The confidence level for VaR and ES calculation (e.g., 0.99 for 99%).

    Returns:
        tuple: VaR and ES values.
    """
    
    # Calculate the exponentially weighted variance
    ew_var = ew_variance(data, lambda_value)
    ew_volatility = np.sqrt(ew_var)

    # Calculate the z-score for the specified confidence level
    z_score = norm.ppf(1 - confidence_level)

    # Value at Risk (VaR) calculation
    VaR = z_score * ew_volatility

    # Expected Shortfall (ES) calculation
    ES = (norm.pdf(z_score) / (1 - confidence_level)) * ew_volatility

    return VaR, ES

# part A
VaR_EW, ES_EW = calculate_var_es_EW(returns, lambda_value, confidence_level)

# Define a function to fit a T-distribution using MLE
def fit_t_distribution(data):
    """
    Fit a T-distribution to the data using Maximum Likelihood Estimation (MLE).
    
    Parameters:
        data (array-like): The data series of returns.
    
    Returns:
        tuple: Estimated parameters (mu, sigma, nu) for the T-distribution.
    """
    
    # Define the negative log-likelihood function for a T-distribution
    def neg_log_likelihood(params):
        mu, sigma, nu = params
        # Prevent sigma and nu from being zero or negative
        if sigma <= 0 or nu <= 2:
            return np.inf
        return -np.sum(t.logpdf(data, df=nu, loc=mu, scale=sigma))
    
    # Initial parameter guesses (mean, standard deviation, degrees of freedom)
    initial_params = [np.mean(data), np.std(data), 5.0]
    
    # Minimize the negative log-likelihood
    result = minimize(neg_log_likelihood, initial_params, bounds=[(None, None), (1e-6, None), (2.01, None)])
    mu, sigma, nu = result.x  # Extract the optimized parameters
    return mu, sigma, nu

# Fit the T-distribution to the returns data
mu, sigma, nu = fit_t_distribution(returns)

# Define the function to calculate VaR and ES for the fitted T-distribution
def calculate_var_es_t(mu, sigma, nu, confidence_level):
    """
    Calculate VaR and ES using a fitted T-distribution.
    
    Parameters:
        mu (float): Mean of the fitted T-distribution.
        sigma (float): Scale (standard deviation) of the fitted T-distribution.
        nu (float): Degrees of freedom for the fitted T-distribution.
        confidence_level (float): The confidence level for VaR and ES calculation (e.g., 0.99 for 99%).
    
    Returns:
        tuple: VaR and ES values.
    """
    # Calculate the quantile for VaR
    var_quantile = t.ppf(1 - confidence_level, df=nu)
    VaR = mu + sigma * var_quantile

    # Calculate Expected Shortfall (ES)
    es_integral = lambda x: x * t.pdf(x, df=nu)
    # ES is calculated as the mean shortfall beyond VaR
    ES = mu + sigma * (t.pdf(var_quantile, df=nu) / (1 - confidence_level)) * (nu + var_quantile**2) / (nu - 1)
    
    return VaR, ES

# part B
VaR_T, ES_T = calculate_var_es_t(mu, sigma, nu, confidence_level)

# Define the function to calculate VaR and ES using Historical Simulation
def calculate_var_es_historical(data, confidence_level):
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES) using Historical Simulation.
    
    Parameters:
        data (array-like): The data series of returns.
        confidence_level (float): The confidence level for VaR and ES calculation (e.g., 0.99 for 99%).
    
    Returns:
        tuple: VaR and ES values.
    """
    # Sort the returns data
    sorted_data = np.sort(data)
    
    # Calculate the index for VaR based on the confidence level
    var_index = int((1 - confidence_level) * len(sorted_data))
    
    # Calculate VaR
    VaR = sorted_data[var_index]
    
    # Calculate ES as the average of all losses beyond the VaR threshold
    ES = np.mean(sorted_data[:var_index])
    
    return -VaR, -ES

# Part C
VaR_HS, ES_HS = calculate_var_es_historical(returns, confidence_level)

# Output
print(f"VaR (Norm distr with an EWMA): {abs(VaR_EW):.4f}")
print(f"ES (Norm distr with an EWMA): {ES_EW:.4f}")

print(f"VaR (MLE fitted T-distribution): {abs(VaR_T):.4f}")
print(f"ES (MLE fitted T-distribution): {ES_T:.4f}")

print(f"VaR (Historical Simulation): {abs(VaR_HS):.4f}")
print(f"ES (Historical Simulation): {abs(ES_HS):.4f}")
