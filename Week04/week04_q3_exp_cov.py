import pandas as pd
import numpy as np

# Load the Portfolio and Daily Prices files
portfolio_file = 'portfolio.csv'
daily_prices_file = 'DailyPrices.csv'

# Load the data into DataFrames
portfolio_df = pd.read_csv(portfolio_file)
daily_prices_df = pd.read_csv(daily_prices_file)

# Calculate daily returns for all assets
daily_returns_df = daily_prices_df.set_index('Date').pct_change().dropna()

# Define the lambda for the exponentially weighted covariance
lambda_value = 0.97

# Define Z-score for the 95% confidence interval
Z_score = 1.645  # For 95% confidence level

# Function to calculate the exponentially weighted covariance matrix
def exponentially_weighted_covariance(returns, lambda_value):
    weights = np.array([(1 - lambda_value) * lambda_value**i for i in range(len(returns))][::-1])
    weighted_returns = returns * np.sqrt(weights[:, np.newaxis])
    return np.cov(weighted_returns, rowvar=False, aweights=weights)

# Calculate the covariance matrix for all assets
cov_matrix = exponentially_weighted_covariance(daily_returns_df.values, lambda_value)

# Extract the portfolios and calculate the returns for each portfolio
# Get unique portfolios from the portfolio_df
portfolios = portfolio_df['Portfolio'].unique()

# Filter out stocks in the portfolio that are not present in the daily returns
available_stocks = daily_returns_df.columns.intersection(portfolio_df['Stock'])

# Filter the portfolio data for available stocks only
filtered_portfolio_df = portfolio_df[portfolio_df['Stock'].isin(available_stocks)]

# Recalculate VaR for each portfolio with filtered stocks
portfolio_var_results = {}

for portfolio in portfolios:
    # Filter the holdings for the current portfolio
    portfolio_holdings = filtered_portfolio_df[filtered_portfolio_df['Portfolio'] == portfolio]
    
    # Filter the daily returns for the stocks in the current portfolio
    stocks = portfolio_holdings['Stock'].values
    holdings = portfolio_holdings['Holding'].values
    
    portfolio_returns = daily_returns_df[stocks]
    portfolio_weights = holdings / holdings.sum()  # Normalize holdings to get weights
    
    # Portfolio variance: w.T * Cov * w
    portfolio_variance = portfolio_weights.T @ cov_matrix[np.ix_(daily_returns_df.columns.isin(stocks), daily_returns_df.columns.isin(stocks))] @ portfolio_weights
    
    # Portfolio standard deviation
    portfolio_std = np.sqrt(portfolio_variance)
    
    # VaR with a 95% confidence interval
    portfolio_var = Z_score * portfolio_std * holdings.sum()  # VaR expressed in $

    # Store the result
    portfolio_var_results[portfolio] = portfolio_var

# Calculate total VaR for all portfolios combined using available stocks
all_stocks = filtered_portfolio_df['Stock'].values
all_holdings = filtered_portfolio_df['Holding'].values

all_portfolio_returns = daily_returns_df[all_stocks]
all_portfolio_weights = all_holdings / all_holdings.sum()

# Total portfolio variance
total_variance = all_portfolio_weights.T @ cov_matrix[np.ix_(daily_returns_df.columns.isin(all_stocks), daily_returns_df.columns.isin(all_stocks))] @ all_portfolio_weights

# Total portfolio standard deviation
total_std = np.sqrt(total_variance)

# Total VaR with a 95% confidence interval
total_var = Z_score * total_std * all_holdings.sum()

# Add total VaR to the results
portfolio_var_results['Total_VaR'] = total_var

# Display the final results
for portfolio, var in portfolio_var_results.items():
    print(f"Holding {portfolio}: VaR = ${abs(var):.2f}")
