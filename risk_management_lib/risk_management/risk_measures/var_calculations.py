# Value at Risk (VaR) calculation methods
from risk_management_lib.risk_management.utils.helper_functions import ew_variance
from scipy.stats import norm, t
import numpy as np

def var_normal(data, confidence_level, lambda_value=0.97):
    """
    Calculate VaR using a normal distribution assumption, given either precomputed volatility
    or raw returns data for EWMA variance calculation.
    
    Parameters:
        data (array-like or float): Returns time series or precomputed volatility.
        confidence_level (float): Confidence level for VaR calculation (e.g., 0.99).
        lambda_value (float): Decay factor for EWMA, used if data is raw returns.
        
    Returns:
        float: Calculated VaR value.
    """
    # Determine if data is already volatility or needs EWMA variance calculation
    if isinstance(data, (float, np.float64)):
        volatility = data  # If data is already volatility
    else:
        # Calculate the exponentially weighted variance from raw returns data
        ew_var = ew_variance(data, lambda_value)
        volatility = np.sqrt(ew_var)  # Convert variance to standard deviation

    # Calculate the z-score for the specified confidence level
    z_score = norm.ppf(1 - confidence_level)
    VaR = z_score * volatility
    return abs(VaR)


def var_t(mu, sigma, nu, confidence_level):
    var_quantile = t.ppf(1 - confidence_level, df=nu)
    return abs(mu + sigma * var_quantile)

def var_historical(returns, confidence_level):
    sorted_returns = np.sort(returns)
    var_index = int((1 - confidence_level) * len(sorted_returns))
    return abs(sorted_returns[var_index])
