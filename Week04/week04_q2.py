import pandas as pd
import numpy as np
from scipy.stats import norm, t
import statsmodels.api as sm
import warnings

# Load the DailyPrices CSV file
daily_prices_file = 'DailyPrices.csv'
daily_prices_df = pd.read_csv(daily_prices_file)

# Define a function for return calculation allowing different methods
def return_calculate(prices, method='arithmetic'):
    """
    Calculate returns based on the method.
    :param prices: Series or DataFrame of prices.
    :param method: The method of return calculation ('arithmetic' or 'logarithmic').
    :return: Returns as per the specified method.
    """
  
    # Convert method to uppercase to ensure consistency
    method = method.lower()

    if method == 'arithmetic':
        returns = prices.pct_change().dropna()  # Arithmetic returns
    elif method == 'logarithmic':
        returns = np.log(prices / prices.shift(1)).dropna()  # Logarithmic returns
    else:
        raise ValueError("Unsupported method. Choose 'arithmetic' or 'logarithmic'.")
    
    return returns

# Select only numeric columns from the DataFrame (skip non-numeric columns)
numeric_columns = daily_prices_df.select_dtypes(include=[np.number])

# Create an empty DataFrame to store the all calculated returns
all_calculated_returns_df = pd.DataFrame()

# Loop through each asset (column) in the DataFrame
for asset in numeric_columns.columns:
    asset_prices = numeric_columns[asset]
    # Calculate the arithmetic returns for each asset
    returns = return_calculate(asset_prices, method='arithmetic')
    
    # Add the returns to the all calculated returns DataFrame
    all_calculated_returns_df[asset] = returns

# Write the all calculated returns to a CSV file
all_calculated_returns_df.to_csv('all_calculated_returns.csv', index=False)

meta_prices = daily_prices_df['META']
# Calculate the arithmetic returns for the 'META' stock
meta_arithmetic_returns = return_calculate(meta_prices, method='arithmetic')

# Calculate the logarithmic returns for the 'META' stock
meta_log_returns = return_calculate(meta_prices, method='logarithmic')

# Remove the mean from the 'META' series
meta_mean_removed = meta_arithmetic_returns - meta_arithmetic_returns.mean()
meta_log_mean_removed = meta_log_returns - meta_log_returns.mean()


# Method 1: Normal Distribution VaR
def var_normal(returns, confidence_level=0.95):
    """
    Calculate VaR using a normal distribution.
    :param returns: Series of returns.
    :param confidence_level: Confidence level for VaR.
    :return: VaR value.
    """
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Calculate the z-score for the given confidence level
    z_score = norm.ppf(1 - confidence_level)
    
    # VaR = Mean - (z_score * Std Deviation)
    var_value = mean_return + z_score * std_return
    return var_value

# Method 2: Exponentially Weighted VaR (EWMA) with λ = 0.94
def var_ewma(returns, lambda_=0.94, confidence_level=0.95):
    """
    Calculate VaR using Exponentially Weighted Moving Average (EWMA).
    :param returns: Series of returns.
    :param lambda_: The smoothing parameter for EWMA.
    :param confidence_level: Confidence level for VaR.
    :return: EWMA VaR value.
    """
    var_ewma = returns.var()
    for ret in returns:
        var_ewma = lambda_ * var_ewma + (1 - lambda_) * ret ** 2
    
    ewma_std = np.sqrt(var_ewma)
    z_score = norm.ppf(1 - confidence_level)
    var_ewma_value = returns.mean() + z_score * ewma_std
    return var_ewma_value

# Method 3: MLE Fitted T-distribution VaR
def var_t_distribution(returns, confidence_level=0.95):
    """
    Calculate VaR using a T-distribution fitted by Maximum Likelihood Estimation (MLE).
    :param returns: Series of returns.
    :param confidence_level: Confidence level for VaR.
    :return: T-distribution VaR value.
    """
    params = t.fit(returns)
    var_t = t.ppf(1 - confidence_level, *params)
    var_t_value = returns.mean() + var_t * returns.std()
    return var_t_value

# Method 4: fitted AR(1) model VaR
def var_ar1(returns, confidence_level=0.95):
    """
    Calculate VaR using an AR(1) model.
    :param returns: Series of returns.
    :param confidence_level: Confidence level for VaR.
    :return: AR(1) VaR value.
    """
    # Assign an integer index explicitly to avoid ARIMA warnings
    returns.index = np.arange(len(returns))
    
    # Suppress warnings related to unsupported index in ARIMA model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Fit an AR(1) model to the returns with integer index
        ar1_model = sm.tsa.ARIMA(returns, order=(1, 0, 0)).fit()

    # Forecasting 1 step ahead (only mean and variance are returned in this case)
    forecast = ar1_model.get_forecast(steps=1)
    forecast_mean = forecast.predicted_mean.iloc[0]
    forecast_std = np.sqrt(forecast.var_pred_mean.iloc[0])

    # Calculate the z-score for the given confidence level
    z_score = norm.ppf(1 - confidence_level)

    # VaR = Forecast Mean - (z_score * Forecast Std Deviation)
    var_ar1_value = forecast_mean + z_score * forecast_std
    return var_ar1_value


# Method 5: Historical Simulation VaR
def var_historical(returns, confidence_level=0.95):
    """
    Calculate VaR using Historical Simulation.
    :param returns: Series of returns.
    :param confidence_level: Confidence level for VaR.
    :return: Historical VaR value.
    """
    # Ensure returns is a 1D Series and flatten any multi-dimensional structure
    returns = pd.Series(np.ravel(returns))  # Flatten the array
    returns = returns.dropna()  # Drop any NaN values
    
    # Sort the returns to get the lowest percentiles
    sorted_returns = returns.sort_values()
    var_index = int((1 - confidence_level) * len(sorted_returns))
    return sorted_returns.iloc[var_index]

# Normal Distribution VaR
var_normal_value = var_normal(meta_mean_removed)
print(f"VaR (Normal Distribution): {var_normal_value * 100:.3f}%")

# EWMA VaR
var_ewma_value = var_ewma(meta_mean_removed)
print(f"VaR (Normal distribution with an Exponentially Weighted variance (λ = 0. 94)): {var_ewma_value * 100:.3f}%")

# T-distribution VaR
var_t_value = var_t_distribution(meta_mean_removed)
print(f"VaR (MLE fitted T distribution): {var_t_value * 100:.3f}%")

# fitted AR(1) model VaR
var_ar1_value = var_ar1(meta_mean_removed)
print(f"VaR (fitted AR(1) model): {var_ar1_value * 100:.3f}%")

# Historical Simulation VaR
var_historical_value = var_historical(meta_mean_removed)
print(f"VaR (Historical Simulation): {var_historical_value * 100:.3f}%")
