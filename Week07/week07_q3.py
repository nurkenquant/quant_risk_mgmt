import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize

# Load the Fama-French 3-factor data and the Carhart Momentum factor data
fama_french_file = 'F-F_Research_Data_Factors_daily.CSV'
momentum_factor_file = 'F-F_Momentum_Factor_daily.CSV'

# Read the data
fama_french_data = pd.read_csv(fama_french_file)
momentum_factor_data = pd.read_csv(momentum_factor_file)

# Clean and preprocess the data
fama_french_data['Date'] = pd.to_datetime(fama_french_data['Date'], format='%Y%m%d')
momentum_factor_data = momentum_factor_data.dropna(subset=['Date'])
momentum_factor_data = momentum_factor_data[momentum_factor_data['Date'].str.isnumeric()]
momentum_factor_data['Date'] = pd.to_datetime(momentum_factor_data['Date'], format='%Y%m%d')

# Filter data for the past 10 years
end_date = fama_french_data['Date'].max()
start_date = end_date - pd.DateOffset(years=10)
fama_french_data_filtered = fama_french_data[(fama_french_data['Date'] >= start_date) & (fama_french_data['Date'] <= end_date)]
momentum_factor_data_filtered = momentum_factor_data[(momentum_factor_data['Date'] >= start_date) & (momentum_factor_data['Date'] <= end_date)]

# Merge the datasets on the Date column and clean column names
factors_data = pd.merge(fama_french_data_filtered, momentum_factor_data_filtered, on='Date', how='inner')
factors_data.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF', 'Mom']
factors_data[['Mkt-RF', 'SMB', 'HML', 'RF', 'Mom']] = factors_data[['Mkt-RF', 'SMB', 'HML', 'RF', 'Mom']] / 100

# Define the list of stocks
stocks = ['AAPL', 'META', 'UNH', 'MA', 'MSFT', 'NVDA', 'HD', 'PFE', 'AMZN', 'BRK-B', 
          'PG', 'XOM', 'TSLA', 'JPM', 'V', 'DIS', 'GOOGL', 'JNJ', 'BAC', 'CSCO']

# Generate synthetic daily returns for each stock
np.random.seed(42)
num_days = len(factors_data)
stock_returns = pd.DataFrame(np.random.normal(0, 0.01, (num_days, len(stocks))), columns=stocks)

# Calculate excess returns for each stock (subtracting the risk-free rate)
excess_returns = stock_returns.subtract(factors_data['RF'].values, axis=0)

# Prepare the factor matrix with a constant term for intercept
factors_matrix = factors_data[['Mkt-RF', 'SMB', 'HML', 'Mom']]
factors_matrix = sm.add_constant(factors_matrix)

# Calculate expected annual returns using 4-factor model
expected_annual_returns = {}
factor_means = factors_matrix.mean()[1:]

for stock in stocks:
    model = sm.OLS(excess_returns[stock], factors_matrix).fit()
    expected_daily_return = model.params.iloc[1:].dot(factor_means) + model.params.iloc[0]
    expected_annual_return = (1 + expected_daily_return) ** 252 - 1
    expected_annual_returns[stock] = expected_annual_return

expected_annual_returns_df = pd.DataFrame(list(expected_annual_returns.items()), columns=['Stock', 'Expected Annual Return']).set_index('Stock')

# Calculate annual covariance matrix of returns
daily_cov_matrix = stock_returns.cov()
annual_cov_matrix = daily_cov_matrix * 252

# Define the risk-free rate
risk_free_rate = 0.05

# Define the optimization objective: Negative Sharpe Ratio
def negative_sharpe(weights, returns, cov_matrix, rf_rate):
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - rf_rate) / portfolio_volatility
    return -sharpe_ratio

# Constraints and bounds for optimization
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(stocks)))
initial_weights = np.array([1/len(stocks)] * len(stocks))

# Perform the optimization
expected_returns = expected_annual_returns_df['Expected Annual Return'].values
result = minimize(negative_sharpe, initial_weights, args=(expected_returns, annual_cov_matrix, risk_free_rate),
                  method='SLSQP', bounds=bounds, constraints=constraints)

# Extract optimized weights and portfolio stats
optimized_weights = result.x
portfolio_return = np.dot(optimized_weights, expected_returns)
portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(annual_cov_matrix, optimized_weights)))
portfolio_sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

optimized_portfolio = pd.DataFrame({'Stock': stocks, 'Weight': optimized_weights}).set_index('Stock')
portfolio_stats = {
    'Expected Portfolio Return': portfolio_return,
    'Portfolio Volatility': portfolio_volatility,
    'Sharpe Ratio': portfolio_sharpe_ratio
}

# Display Expected Annual Returns
print("Expected Annual Returns:")
print(expected_annual_returns_df.round(4))
print("\n")

# Display the Annual Covariance Matrix with rounded values
print("Annual Covariance Matrix of Stock Returns:")
print(annual_cov_matrix.round(4))
print("\n")

# Display Optimized Portfolio Weights with non-zero weights only
print("Optimized Portfolio Weights (Non-zero weights):")
non_zero_weights = optimized_portfolio[optimized_portfolio['Weight'] > 0]
print(non_zero_weights.round(4))
print("\n")

# Display Portfolio Statistics with rounded values
print("Portfolio Statistics:")
for key, value in portfolio_stats.items():
    print(f"{key}: {value:.4f}")

