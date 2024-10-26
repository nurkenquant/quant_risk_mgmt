import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import os
import pandas as pd
import numpy as np
from risk_management_lib.risk_management.utils.helper_functions import return_calculate, ew_variance
from risk_management_lib.risk_management.risk_measures.var_calculations import var_normal

# Load data
portfolio = pd.read_csv('portfolio.csv')
prices = pd.read_csv('DailyPrices.csv')

# parameters
confidence_level = 0.99
lambda_value = 0.97

# Step 1: Calculate Arithmetic Returns
returns = return_calculate(prices, method='arithmetic')

# Step 2: Calculate Asset Values and Weights for Each Portfolio
def calculate_portfolio_weights(portfolio, prices_df):
    """
    Calculate the asset value and weights for each stock in the portfolio.
    """
    latest_prices = prices_df.iloc[-1]
    asset_values = {}
    total_value = 0
    
    for _, row in portfolio.iterrows():
        stock = row['Stock']
        holding = row['Holding']
        if stock in latest_prices.index:
            asset_value = holding * latest_prices[stock]
            asset_values[stock] = asset_value
            total_value += asset_value
    
    # Calculate weights as the proportion of each asset's value to the total portfolio value
    weights = {stock: value / total_value for stock, value in asset_values.items()}
    return weights

# Step 3: Calculate Weighted Portfolio Returns Based on Asset Values
def get_portfolio_weighted_returns(portfolio, returns_df, prices_df):
    """
    Calculate weighted returns for the portfolio based on asset values.
    """
    weights = calculate_portfolio_weights(portfolio, prices_df)
    weighted_returns = pd.Series(0, index=returns_df.index)
    
    for stock, weight in weights.items():
        if stock in returns_df.columns:
            weighted_returns += weight * returns_df[stock]
    
    return weighted_returns

# Calculate portfolio returns for each portfolio
portfolio_a = portfolio[portfolio['Portfolio'] == 'A']
portfolio_b = portfolio[portfolio['Portfolio'] == 'B']
portfolio_c = portfolio[portfolio['Portfolio'] == 'C']

portfolio_a_returns = get_portfolio_weighted_returns(portfolio_a, returns, prices)
portfolio_b_returns = get_portfolio_weighted_returns(portfolio_b, returns, prices)
portfolio_c_returns = get_portfolio_weighted_returns(portfolio_c, returns, prices)

# Step 4: Calculate EWMA Covariance Matrix and Portfolio Volatility
# Stack portfolio returns into a DataFrame
portfolio_returns_df = pd.DataFrame({
    'Portfolio_A': portfolio_a_returns,
    'Portfolio_B': portfolio_b_returns,
    'Portfolio_C': portfolio_c_returns
})

# Calculate the EWMA covariance matrix for all portfolio returns
ew_cov_matrix = ew_variance(portfolio_returns_df, lambda_value)

# Calculate portfolio volatility as the square root of the diagonal of the covariance matrix
volatility_a = np.sqrt(ew_cov_matrix.loc['Portfolio_A', 'Portfolio_A'])
volatility_b = np.sqrt(ew_cov_matrix.loc['Portfolio_B', 'Portfolio_B'])
volatility_c = np.sqrt(ew_cov_matrix.loc['Portfolio_C', 'Portfolio_C'])

# Calculate VaR assuming normal distribution at 99% confidence level
var_a = var_normal(volatility_a, confidence_level, lambda_value)
var_b = var_normal(volatility_b, confidence_level, lambda_value)
var_c = var_normal(volatility_c, confidence_level, lambda_value)

# Step 5: Sum Individual Portfolio VaRs for Total VaR
total_var = var_a + var_b + var_c

# Display Results
print(f"VaR for Portfolio A ($): {var_a}")
print(f"VaR for Portfolio B ($): {var_b}")
print(f"VaR for Portfolio C ($): {var_c}")
print(f"Total VaR ($): {total_var}")
