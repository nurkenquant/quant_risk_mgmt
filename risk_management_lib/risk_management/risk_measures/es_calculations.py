# Expected Shortfall (ES) calculation methods
from risk_management_lib.risk_management.utils.helper_functions import ew_variance
from scipy.stats import norm, t
import numpy as np

def es_normal(returns, confidence_level, lambda_value):
    # Calculate the exponentially weighted variance
    ew_var = ew_variance(returns, lambda_value)
    ew_volatility = np.sqrt(ew_var)

    # Calculate the z-score for the specified confidence level
    z_score = norm.ppf(1 - confidence_level)

    # Expected Shortfall (ES) calculation
    ES = (norm.pdf(z_score) / (1 - confidence_level)) * ew_volatility
    return ES

def es_t(mu, sigma, nu, confidence_level):
    var_quantile = t.ppf(1 - confidence_level, df=nu)
    return mu + sigma * (t.pdf(var_quantile, df=nu) / (1 - confidence_level)) * (nu + var_quantile**2) / (nu - 1)

def es_historical(returns, confidence_level):
    sorted_returns = np.sort(returns)
    var_index = int((1 - confidence_level) * len(sorted_returns))
    return abs(np.mean(sorted_returns[:var_index]))
