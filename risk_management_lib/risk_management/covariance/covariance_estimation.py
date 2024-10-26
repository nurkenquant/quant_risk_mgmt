# Covariance estimation methods
import numpy as np

def sample_covariance(returns):
    return np.cov(returns, rowvar=False)

def ewma_covariance(returns, lambda_value=0.97):
    m = len(returns)
    weights = np.array([(1 - lambda_value) * (lambda_value ** (m - i - 1)) for i in range(m)])
    weights /= weights.sum()
    adjusted_returns = returns - np.dot(weights, returns)
    ewma_cov = np.dot(weights * adjusted_returns.T, adjusted_returns)
    return ewma_cov
