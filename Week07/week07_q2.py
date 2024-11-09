import pandas as pd
import numpy as np
from scipy.stats import norm

# Load the CSV files
daily_prices = pd.read_csv('DailyPrices.csv')
portfolio_file = pd.read_csv('problem2.csv')

# Convert date column to datetime
daily_prices['Date'] = pd.to_datetime(daily_prices['Date'])

# Sort by date to ensure returns are calculated chronologically
daily_prices = daily_prices.sort_values(by='Date')

# Calculate daily returns for AAPL
daily_prices['AAPL_Return'] = daily_prices['AAPL'].pct_change()

# Drop NaN values resulted from the pct_change
aapl_returns = daily_prices['AAPL_Return'].dropna()

# Fit a normal distribution to the AAPL returns assuming a mean return of 0
std_dev = aapl_returns.std()

# Parameters
current_price = 165  # current AAPL price
days_ahead = 10      # number of days to simulate
num_simulations = 10000  # number of simulations
dividend = 1.00      # Dividend amount
dividend_date = pd.to_datetime("2023-03-15")  # Dividend payment date

# Simulate 10-day returns and apply them to the current price
np.random.seed(0)
simulated_returns = np.random.normal(0, std_dev, (num_simulations, days_ahead))
simulated_prices = current_price * np.exp(simulated_returns.cumsum(axis=1))

# Adjust for dividend payment on March 15, 2023
dividend_adjustment_day = (dividend_date - pd.to_datetime("2023-03-03")).days
simulated_prices[:, dividend_adjustment_day:] -= dividend

# Calculate Mean, VaR (95%), and Expected Shortfall (95%) based on the end-of-period prices
final_prices = simulated_prices[:, -1]  # prices at the end of the 10-day period

# Mean of the simulated final prices
mean_price = final_prices.mean()

# Value at Risk (VaR) at 95% confidence level
var_95 = np.percentile(final_prices, 5)

# Expected Shortfall (ES) at 95% confidence level (mean of the worst 5% outcomes)
es_95 = final_prices[final_prices <= var_95].mean()

# Convert to dollar losses from the initial price
mean_loss = current_price - mean_price
var_loss = current_price - var_95
es_loss = current_price - es_95

# Calculate the value of the options in each portfolio based on simulated prices
# and then determine the VaR and ES for each portfolio

# Get the unique portfolios from the problem2 file to analyze each one separately
portfolios = portfolio_file['Portfolio'].unique()

# Initialize dictionary to store VaR and ES for each portfolio
portfolio_risk_metrics = []

# Loop through each portfolio, apply simulated returns, and calculate risk metrics
for portfolio in portfolios:
    # Filter for the specific portfolio
    portfolio_data = portfolio_file[portfolio_file['Portfolio'] == portfolio]
    
    # Calculate the portfolio's value based on simulated final prices
    portfolio_values = []
    
    for price in final_prices:
        # Calculate option payoff based on current simulated price and portfolio details
        portfolio_value = 0
        for _, row in portfolio_data.iterrows():
            strike_price = row['Strike']
            holding = row['Holding']
            option_type = row['OptionType']
            
            # Calculate payoff depending on option type
            if option_type == 'Call':
                payoff = max(price - strike_price, 0) * holding
            elif option_type == 'Put':
                payoff = max(strike_price - price, 0) * holding
            else:
                continue
            
            # Sum up all payoffs for the portfolio value at this price
            portfolio_value += payoff
        
        # Append the calculated portfolio value for this simulated price
        portfolio_values.append(portfolio_value)
    
    # Convert to numpy array for easy calculations
    portfolio_values = np.array(portfolio_values)
    
    # Calculate VaR and ES
    portfolio_mean_loss = portfolio_values.mean()
    portfolio_var_95 = np.percentile(portfolio_values, 5)
    portfolio_es_95 = portfolio_values[portfolio_values <= portfolio_var_95].mean()
    
    # Store the results for this portfolio
    portfolio_risk_metrics.append({
        'Portfolio': portfolio,
        'Simulated Mean Loss': portfolio_mean_loss,  # as a loss from the initial
        'Simulated VaR 95%': portfolio_var_95,
        'Simulated ES 95%': portfolio_es_95
    })

# Convert results to DataFrame for display
portfolio_risk_metrics_df = pd.DataFrame(portfolio_risk_metrics)

# Delta-Normal VaR and ES Calculation

# Define Black-Scholes d1 calculation function adjusted for dividends
def calculate_d1(S, K, T, r, sigma, q):
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

# Given parameters for Delta-Normal VaR
T = (pd.to_datetime("2023-04-21") - pd.to_datetime("2023-03-03")).days / 365  # Time to expiration in years
r = 0.0425  # Risk-free rate
q = dividend / current_price  # Dividend yield
sigma = std_dev * np.sqrt(252)  # Annualized volatility

# Initialize list for storing Delta-Normal results
delta_normal_results = []

# Loop through each portfolio for Delta-Normal VaR and ES calculations
for portfolio in portfolios:
    portfolio_data = portfolio_file[portfolio_file['Portfolio'] == portfolio]
    portfolio_value_change = 0  # sum of delta-adjusted changes
    
    for _, row in portfolio_data.iterrows():
        S = current_price  # Current AAPL price
        K = row['Strike']  # Strike price
        holding = row['Holding']
        option_type = row['OptionType']
        
        # Calculate d1 with dividend adjustment
        d1 = calculate_d1(S, K, T, r, sigma, q)
        
        # Calculate delta based on option type
        if option_type == 'Call':
            delta = norm.cdf(d1) * holding
        elif option_type == 'Put':
            delta = (norm.cdf(d1) - 1) * holding
        else:
            continue
        
        # Calculate dollar value impact (delta * price change)
        portfolio_value_change += delta * S * std_dev * np.sqrt(days_ahead)
    
    # Convert results to dollar losses
    portfolio_var_95_delta = -1.645 * portfolio_value_change  # VaR at 95%
    portfolio_es_95_delta = -2.06 * portfolio_value_change  # ES approx. at 95%
    
    delta_normal_results.append({
        'Portfolio': portfolio,
        'Delta-Normal VaR 95%': portfolio_var_95_delta,
        'Delta-Normal ES 95%': portfolio_es_95_delta
    })

# Convert results to DataFrame for display
delta_normal_results_df = pd.DataFrame(delta_normal_results)

# Combine both DataFrames for a single table output
# Merging based on the 'Portfolio' column

# Merge the two DataFrames on 'Portfolio'
combined_df = pd.merge(portfolio_risk_metrics_df, delta_normal_results_df, on='Portfolio')

# Print the combined DataFrame
print(combined_df)
