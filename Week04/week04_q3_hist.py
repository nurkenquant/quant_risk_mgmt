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

# Extract the portfolios and calculate the returns for each portfolio
# Get unique portfolios from the portfolio_df
portfolios = portfolio_df['Portfolio'].unique()

# Filter out stocks in the portfolio that are not present in the daily returns
available_stocks = daily_returns_df.columns.intersection(portfolio_df['Stock'])

# Filter the portfolio data for available stocks only
filtered_portfolio_df = portfolio_df[portfolio_df['Stock'].isin(available_stocks)]

# Recalculate VaR using historical simulation method for each portfolio
portfolio_var_results_hist = {}

for portfolio in portfolios:
    # Filter the holdings for the current portfolio
    portfolio_holdings = filtered_portfolio_df[filtered_portfolio_df['Portfolio'] == portfolio]
    
    # Filter the daily returns for the stocks in the current portfolio
    stocks = portfolio_holdings['Stock'].values
    holdings = portfolio_holdings['Holding'].values
    
    portfolio_returns = daily_returns_df[stocks]
    portfolio_weights = holdings / holdings.sum()  # Normalize holdings to get weights
    
    # Calculate the weighted portfolio returns (sum of weighted individual stock returns)
    weighted_portfolio_returns = portfolio_returns.dot(portfolio_weights)
    
    # Sort the portfolio returns and select the 5th percentile (for 95% VaR)
    portfolio_var_hist = np.percentile(weighted_portfolio_returns, 5)  # 5th percentile
    
    # VaR in dollars
    portfolio_value = holdings.sum()
    portfolio_var_hist_dollar = portfolio_var_hist * portfolio_value

    # Store the result
    portfolio_var_results_hist[portfolio] = portfolio_var_hist_dollar

# Calculate total VaR for all portfolios combined using available stocks
all_stocks = filtered_portfolio_df['Stock'].values
all_holdings = filtered_portfolio_df['Holding'].values

all_portfolio_returns = daily_returns_df[all_stocks]
all_portfolio_weights = all_holdings / all_holdings.sum()

# Total portfolio returns (weighted sum of all available stocks)
weighted_all_portfolio_returns = all_portfolio_returns.dot(all_portfolio_weights)

# Total portfolio VaR (5th percentile of total portfolio returns)
total_var_hist = np.percentile(weighted_all_portfolio_returns, 5)
total_var_hist_dollar = total_var_hist * all_holdings.sum()

# Add total VaR to the results
portfolio_var_results_hist['Total_VaR'] = total_var_hist_dollar

# Display the final results
for portfolio, var in portfolio_var_results_hist.items():
    print(f"Holding {portfolio}: Historical Simulation VaR = ${abs(var):.2f}")
