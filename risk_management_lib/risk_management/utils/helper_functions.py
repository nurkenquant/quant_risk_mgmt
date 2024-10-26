# Helper functions
from scipy.stats import t
from scipy.optimize import minimize
import numpy as np
import pandas as pd

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

def fit_generalized_t(data):
    """
    Fit a T-distribution to the data using MLE.

    Parameters:
        data (array-like): The data series of returns.

    Returns:
        tuple: Estimated parameters (mu, sigma, nu) for the T-distribution.
    """
    def neg_log_likelihood(params):
        mu, sigma, nu = params
        if sigma <= 0 or nu <= 2:
            return np.inf
        return -np.sum(t.logpdf(data, df=nu, loc=mu, scale=sigma))

    initial_params = [np.mean(data), np.std(data), 5.0]
    result = minimize(neg_log_likelihood, initial_params, bounds=[(None, None), (1e-6, None), (2.01, None)])
    mu, sigma, nu = result.x
    return mu, sigma, nu

def return_calculate(prices, method='arithmetic'):
    """
    Calculate returns based on the method.
    
    Parameters:
        prices (Series or DataFrame): The series or DataFrame of prices.
        method (str): The method of return calculation ('arithmetic' or 'logarithmic').
    
    Returns:
        DataFrame: Returns calculated as per the specified method.
    """
    method = method.lower()
    
    # Ensure all data is numeric and drop non-numeric columns
    prices = prices.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').dropna()
    
    if method == 'arithmetic':
        returns = prices.pct_change().dropna()  # Arithmetic returns
    elif method == 'logarithmic':
        returns = np.log(prices / prices.shift(1)).dropna()  # Logarithmic returns
    else:
        raise ValueError("Unsupported method. Choose 'arithmetic' or 'logarithmic'.")
    return returns
