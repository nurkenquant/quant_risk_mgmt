import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime

# Parameters
S = 165                # Current Stock Price
r = 5.25 /100             # Risk-Free Rate (5.25%)
q = 0.53 / 100             # Continuously Compounding Coupon (0.53%)

# Dates for time to maturity calculation
current_date = datetime.strptime("2023-03-03", "%Y-%m-%d")
expiration_date = datetime.strptime("2023-03-17", "%Y-%m-%d")

# Calculate time to maturity in years
days_to_maturity = (expiration_date - current_date).days
T = days_to_maturity / 365  # Time to Maturity in Years

# Implied volatilities from 10% to 80%
implied_volatilities = np.linspace(0.1, 0.8, 100)

# Strike price assumption (assume it is at-the-money initially)
K = S  # assuming strike price is the same as stock price

# Black-Scholes formula for Call and Put options
def black_scholes_call_put(S, K, r, T, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return call, put

# Calculate call and put option values for each implied volatility
call_values = []
put_values = []

for sigma in implied_volatilities:
    call, put = black_scholes_call_put(S, K, r, T, sigma, q)
    call_values.append(call)
    put_values.append(put)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(implied_volatilities * 100, call_values, label='Call Option Value')
plt.plot(implied_volatilities * 100, put_values, label='Put Option Value')
plt.xlabel("Implied Volatility (%)")
plt.ylabel("Option Value ($)")
plt.title("Effect of Implied Volatility on Call and Put Option Values")
plt.legend()
plt.grid()
plt.show()
