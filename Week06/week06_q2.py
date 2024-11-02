import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from datetime import datetime

# Load the AAPL options data
file_path = 'AAPL_Options.csv'
aapl_options = pd.read_csv(file_path)

# Parameters
current_price = 170.15  # Current AAPL price
current_date = datetime.strptime("10/30/2023", "%m/%d/%Y")
risk_free_rate = 5.25 / 100  # Risk-free rate
dividend_rate = 0.57 / 100  # Dividend rate


# Calculate days to expiration for each option
aapl_options['Expiration'] = pd.to_datetime(aapl_options['Expiration'], format='%m/%d/%Y')
aapl_options['DaysToExpiration'] = (aapl_options['Expiration'] - current_date).dt.days

# Function to calculate d1 and d2 for Black-Scholes model
def d1(S, K, T, r, sigma, q):
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma, q):
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

# Black-Scholes formulas for Call and Put prices
def black_scholes_call(S, K, T, r, sigma, q):
    return S * np.exp(-q * T) * norm.cdf(d1(S, K, T, r, sigma, q)) - K * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma, q))

def black_scholes_put(S, K, T, r, sigma, q):
    return K * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma, q)) - S * np.exp(-q * T) * norm.cdf(-d1(S, K, T, r, sigma, q))

# Function to calculate implied volatility
def implied_volatility(option_type, S, K, T, r, market_price, q):
    # Define objective function for root-finding
    if option_type == 'Call':
        price_function = lambda sigma: black_scholes_call(S, K, T, r, sigma, q) - market_price
    else:  # Put
        price_function = lambda sigma: black_scholes_put(S, K, T, r, sigma, q) - market_price
    
    # Initial bounds for volatility
    try:
        implied_vol = brentq(price_function, 0.01, 5.0, maxiter=500)  # Attempt to find a root
    except ValueError:
        implied_vol = np.nan  # If no solution, assign NaN
    return implied_vol

# Calculate implied volatility for each option
aapl_options['ImpliedVolatility'] = aapl_options.apply(
    lambda row: implied_volatility(
        row['Type'], current_price, row['Strike'], row['DaysToExpiration'] / 365, 
        risk_free_rate, row['Last Price'], dividend_rate
    ), axis=1
)

# Plotting implied volatility vs. strike price for Calls and Puts
calls = aapl_options[aapl_options['Type'] == 'Call']
puts = aapl_options[aapl_options['Type'] == 'Put']

plt.figure(figsize=(12, 6))
plt.plot(calls['Strike'], calls['ImpliedVolatility'], label='Call Options', marker='o')
plt.plot(puts['Strike'], puts['ImpliedVolatility'], label='Put Options', marker='x')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility vs. Strike Price for AAPL Options')
plt.legend()
plt.grid(True)
plt.show()
