import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.ar_model import AutoReg

# Load CSV files for problem3 and DailyPrices
options_df = pd.read_csv('problem3.csv')
daily_prices_df = pd.read_csv('DailyPrices.csv')

# Parameters
current_aapl_price = 170.15  # Current AAPL price
risk_free_rate = 5.25 / 100      # 5.25%
dividend_rate = 0.57 / 100       # 0.57%
current_date = pd.to_datetime("2023-10-30")  # current date

# Define range of AAPL prices for plotting
underlying_prices = np.linspace(100, 250, 100)

# Calculate implied volatility as the standard deviation of log returns
daily_prices_df['Date'] = pd.to_datetime(daily_prices_df['Date'])
daily_prices_df.sort_values('Date', inplace=True)
daily_prices_df['AAPL_Log_Returns'] = np.log(daily_prices_df['AAPL']).diff()
implied_volatility = daily_prices_df['AAPL_Log_Returns'].std() * np.sqrt(252)  # Annualized implied volatility

# Black-Scholes formula for call and put prices with dividend yield
def black_scholes_call(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price

# Calculate portfolio value for a given underlying price using Black-Scholes prices with implied volatility and dividend yield
def portfolio_value(portfolio, price, time_to_expiry):
    value = 0
    for _, option in portfolio.iterrows():
        strike = option['Strike']
        option_type = option['OptionType']
        holding = option['Holding']
        
        # Initialize payoff with zero in case of an undefined option type
        payoff = 0
        if option_type == 'Call':
            payoff = black_scholes_call(price, strike, time_to_expiry, risk_free_rate, implied_volatility, dividend_rate)
        elif option_type == 'Put':
            payoff = black_scholes_put(price, strike, time_to_expiry, risk_free_rate, implied_volatility, dividend_rate)

        value += holding * payoff
    return value

# Calculate time to expiration for each portfolio in years
options_df['ExpirationDate'] = pd.to_datetime(options_df['ExpirationDate'])
options_df['DaysToExpiry'] = (options_df['ExpirationDate'] - current_date).dt.days / 365  # Convert days to years

# Calculate initial portfolio values at current date and current AAPL price
initial_portfolio_values = {}
for name, portfolio in options_df.groupby('Portfolio'):
    time_to_expiry = portfolio['DaysToExpiry'].iloc[0] 
    initial_portfolio_values[name] = portfolio_value(portfolio, current_aapl_price, time_to_expiry)

# Plot portfolio values over range of AAPL prices and calculate P&L
plt.figure(figsize=(12, 8))
for name, portfolio in options_df.groupby('Portfolio'):
    time_to_expiry = portfolio['DaysToExpiry'].iloc[0]  # Assuming same expiry within each portfolio
    values = [portfolio_value(portfolio, price, time_to_expiry) for price in underlying_prices]
    PL = np.array(values) - initial_portfolio_values[name]  # Calculate P&L as change from initial value
    plt.plot(underlying_prices, PL, label=name)

# Plotting details
plt.title("P&L for Portfolios over a Range of Underlying AAPL Prices")
plt.xlabel("Underlying AAPL Price")
plt.ylabel("P&L")
plt.legend()
plt.grid(True)
plt.show()

# Function to simulate AR(1) process
def ar1(demeaned_returns, current_price, days_ahead=10):
    demeaned_returns = demeaned_returns.reset_index(drop=True)
    
    # Fit AR(1) model
    model = AutoReg(demeaned_returns, lags=1)
    model_fit = model.fit()

    # Extract AR(1) parameters
    phi = model_fit.params['AAPL_Log_Returns.L1']  # AR(1) coefficient
    mu = model_fit.params['const']  # Intercept
    sigma = model_fit.sigma2**0.5   # Standard deviation of residuals

    # Simulate future returns
    np.random.seed(0)  # For reproducibility
    simulated_returns = [demeaned_returns.iloc[-1]]
    for _ in range(1, days_ahead):
        simulated_return = mu + phi * simulated_returns[-1] + np.random.normal(0, sigma)
        simulated_returns.append(simulated_return)

    # Forecast prices
    simulated_returns = np.array(simulated_returns)
    simulated_prices = current_price * np.exp(simulated_returns.cumsum())
    return simulated_prices

# Simulate 10 days ahead and calculate Mean, VaR, and ES
aapl_log_returns_demeaned = daily_prices_df['AAPL_Log_Returns'].dropna() - daily_prices_df['AAPL_Log_Returns'].mean()
simulated_prices = ar1(aapl_log_returns_demeaned, current_aapl_price)
mean_price = simulated_prices.mean()
VaR = np.percentile(simulated_prices, 5)  # 5th percentile for VaR at 95% confidence
ES = simulated_prices[simulated_prices <= VaR].mean()  # ES is mean of values below VaR

# Output results
print(f"Mean Price: {mean_price:.2f}")
print(f"Value at Risk (VaR 95%): {VaR:.2f}")
print(f"Expected Shortfall (ES 95%): {ES:.2f}")
