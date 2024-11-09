import math
from scipy.stats import norm
from datetime import datetime
import numpy as np
import pandas as pd

# GBSM function to calculate option price
def gbsm(call, underlying, strike, ttm, rf, b, ivol):
    """
    Generalized Black-Scholes-Merton (GBSM) model for option pricing.
    
    Parameters:
    call (bool): True for call option, False for put option
    underlying (float): Current stock price
    strike (float): Strike price
    ttm (float): Time to maturity in years
    rf (float): Risk-free rate (as a decimal)
    b (float): Cost of carry (continuously compounded dividend yield)
    ivol (float): Implied volatility (as a decimal)
    
    Returns:
    float: Option price
    """
    d1 = (math.log(underlying / strike) + (b + ivol**2 / 2) * ttm) / (ivol * math.sqrt(ttm))
    d2 = d1 - ivol * math.sqrt(ttm)
    
    if call:
        # Call option formula
        return underlying * math.exp((b - rf) * ttm) * norm.cdf(d1) - strike * math.exp(-rf * ttm) * norm.cdf(d2)
    else:
        # Put option formula
        return strike * math.exp(-rf * ttm) * norm.cdf(-d2) - underlying * math.exp((b - rf) * ttm) * norm.cdf(-d1)

# Function for closed form greeks calculation
def closed_form_greeks(call, underlying, strike, ttm, rf, b, ivol):
    """
    Calculate the closed-form Greeks for the GBSM (European options with continuous dividends).

    Parameters:
    call (bool): True for call option, False for put option
    underlying (float): Current stock price
    strike (float): Strike price
    ttm (float): Time to maturity in years
    rf (float): Risk-free rate (as a decimal)
    b (float): Cost of carry (continuously compounded dividend yield)
    ivol (float): Implied volatility (as a decimal)

    Returns:
    dict: Closed-form Greeks - Delta, Gamma, Theta, Vega, Rho
    """
    d1 = (math.log(underlying / strike) + (b + ivol**2 / 2) * ttm) / (ivol * math.sqrt(ttm))
    d2 = d1 - ivol * math.sqrt(ttm)
    
    # Delta
    delta = math.exp((b - rf) * ttm) * norm.cdf(d1) if call else math.exp((b - rf) * ttm) * (norm.cdf(d1) - 1)
    
    # Gamma
    gamma = math.exp((b - rf) * ttm) * norm.pdf(d1) / (underlying * ivol * math.sqrt(ttm))
    
    # Theta
    if call:
        theta = (-underlying * math.exp((b - rf) * ttm) * norm.pdf(d1) * ivol / (2 * math.sqrt(ttm))
                 - (b - rf) * underlying * math.exp((b - rf) * ttm) * norm.cdf(d1)
                 - rf * strike * math.exp(-rf * ttm) * norm.cdf(d2))
    else:
        theta = (-underlying * math.exp((b - rf) * ttm) * norm.pdf(d1) * ivol / (2 * math.sqrt(ttm))
                 + (b - rf) * underlying * math.exp((b - rf) * ttm) * norm.cdf(-d1)
                 + rf * strike * math.exp(-rf * ttm) * norm.cdf(-d2))
    
    # Vega
    vega = underlying * math.exp((b - rf) * ttm) * norm.pdf(d1) * math.sqrt(ttm)
    
    # Rho
    if call:
        rho = strike * ttm * math.exp(-rf * ttm) * norm.cdf(d2)
    else:
        rho = -strike * ttm * math.exp(-rf * ttm) * norm.cdf(-d2)
    
    # Carry Rho
    if call:
        c_rho = underlying * ttm * math.exp((b - rf) * ttm) * norm.cdf(d1)
    else:
        c_rho = -underlying * ttm * math.exp((b - rf) * ttm) * norm.cdf(-d1)
        
    return {
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta / 365,  # Theta per day
        "Vega": vega / 100,  # Vega per 1% change in volatility
        "Rho": rho / 100,  # Rho per 1% change in interest rate
        "Carry Rho": c_rho / 100  # Carry Rho per 1% change in cost of carry
    }

# Function for finite difference greeks calculation
def finite_difference_greeks(call, underlying, strike, ttm, rf, b, ivol, epsilon=1e-5):
    """
    Calculate Greeks using finite difference method for a given option.

    Parameters:
    call (bool): True for call option, False for put option
    underlying (float): Current stock price
    strike (float): Strike price
    ttm (float): Time to maturity in years
    rf (float): Risk-free rate (as a decimal)
    b (float): Cost of carry (continuously compounded dividend yield)
    ivol (float): Implied volatility (as a decimal)
    epsilon (float): Small change for finite difference

    Returns:
    dict: Approximated Greeks - Delta, Gamma, Theta, Vega, Rho
    """
    # Base price for option
    price_base = gbsm(call, underlying, strike, ttm, rf, b, ivol)

    # Delta approximation
    price_up_delta = gbsm(call, underlying + epsilon, strike, ttm, rf, b, ivol)
    delta = (price_up_delta - price_base) / epsilon

    # Gamma approximation
    price_down_delta = gbsm(call, underlying - epsilon, strike, ttm, rf, b, ivol)
    gamma = (price_up_delta - 2 * price_base + price_down_delta) / (epsilon ** 2)

    # Theta approximation (1 day decrement)
    price_up_theta = gbsm(call, underlying, strike, ttm - epsilon / 365, rf, b, ivol)
    theta = (price_up_theta - price_base) / (-epsilon)  # Negative because time to expiration decreases

    # Vega approximation
    price_up_vega = gbsm(call, underlying, strike, ttm, rf, b, ivol + epsilon)
    vega = (price_up_vega - price_base) / epsilon

    # Rho approximation
    price_up_rho = gbsm(call, underlying, strike, ttm, rf + epsilon, b, ivol)
    rho = (price_up_rho - price_base) / epsilon
    
    # Carry Rho approximation
    price_up_carry_rho = gbsm(call, underlying, strike, ttm, rf, b + epsilon, ivol)
    carry_rho = (price_up_carry_rho - price_base) / epsilon

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta / 365,  # Theta per day
        "Vega": vega / 100,  # Vega per 1% change in volatility
        "Rho": rho / 100,  # Rho per 1% change in interest rate
        "Carry Rho": carry_rho / 100  # Carry Rho per 1% change in cost of carry
    }

# Parameters
underlying_price = 151.03  # Current stock price
strike_price = 165  # Strike price
current_date = "03/13/2022"  # Current date
expiration_date = "04/15/2022"  # Options expiration date
risk_free_rate = 4.25 / 100  # Risk-free rate
dividend_yield = 0.53 / 100  # Continuously compounding coupon
implied_volatility = 20 / 100  # Implied volatility

# Calculate time to maturity (in years)
current_date_dt = datetime.strptime(current_date, "%m/%d/%Y")
expiration_date_dt = datetime.strptime(expiration_date, "%m/%d/%Y")
time_to_maturity = (expiration_date_dt - current_date_dt).days / 365.0

# Calculate closed-form Greeks
closed_form_call_greeks = closed_form_greeks(True, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility)
closed_form_put_greeks = closed_form_greeks(False, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility)

# Calculate finite difference Greeks for both call and put options
finite_diff_call_greeks = finite_difference_greeks(True, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility)
finite_diff_put_greeks = finite_difference_greeks(False, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility)

def compare_greeks(closed_form, finite_diff, option_type="Call"):
    """
    Compares the closed-form and finite difference Greeks for a given option type and prints them.

    Parameters:
    closed_form (dict): Closed-form Greeks for the option.
    finite_diff (dict): Finite difference Greeks for the option.
    option_type (str): "Call" or "Put" to specify the option type.
    """
    print(f"\n{option_type} Option Greek Comparison:")
    print(f"{'Greek':<12}{'Closed-form':<15}{'Finite Difference':<20}{'Difference':<15}")
    print("-" * 60)

    for greek in closed_form.keys():
        closed_val = closed_form[greek]
        finite_val = finite_diff[greek]
        diff = closed_val - finite_val
        print(f"{greek:<12}{closed_val:<15.4f}{finite_val:<20.4f}{diff:<15.4f}")

# Calculate the GBSM call and put options 
call_gbsm = gbsm(True, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility)
put_gbsm = gbsm(False, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility)

# Call the compare function to print the comparison for both call and put options
compare_greeks(closed_form_call_greeks, finite_diff_call_greeks, "Call")
compare_greeks(closed_form_put_greeks, finite_diff_put_greeks, "Put")

# Implement the binomial tree valuation for American options with discrete dividends
def binomial_tree_american(call, underlying, strike, ttm, rf, b, ivol, N, dividend_date, dividend_amount):
    """
    Binomial tree valuation for American options, considering a discrete dividend.
    
    Parameters:
    call (bool): True for call option, False for put option
    underlying (float): Current stock price
    strike (float): Strike price
    ttm (float): Time to maturity in years
    rf (float): Risk-free rate (as a decimal)
    b (float): Cost of carry (continuously compounded dividend yield)
    ivol (float): Implied volatility (as a decimal)
    N (int): Number of time steps in the binomial tree
    dividend_date (float): Time to dividend date (in years)
    dividend_amount (float): Dividend amount paid on dividend_date

    Returns:
    float: Option price
    """
    dt = ttm / N  # Time step
    u = np.exp(ivol * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    pu = (np.exp(b * dt) - d) / (u - d)  # Probability of up move
    pd = 1 - pu  # Probability of down move
    df = np.exp(-rf * dt)  # Discount factor per step
    z = 1 if call else -1  # Direction multiplier for payoff
    
    # Adjust underlying price for dividend
    if dividend_date < ttm:
        underlying_ex_div = underlying - dividend_amount * np.exp(-rf * dividend_date / dt)
    else:
        underlying_ex_div = underlying

    # Set up the tree for option prices at maturity
    option_values = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        price = underlying_ex_div * (u ** i) * (d ** (N - i))
        option_values[i, N] = max(0, z * (price - strike))

    # Backward induction for option value at each node
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            option_value = df * (pu * option_values[i + 1, j + 1] + pd * option_values[i, j + 1])
            price = underlying_ex_div * (u ** i) * (d ** (j - i))
            option_values[i, j] = max(option_value, z * (price - strike))  # American option early exercise

    return option_values[0, 0]

call_gbsm = gbsm(True, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility)
put_gbsm = gbsm(False, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility)


# Parameters for binomial tree valuation
N_steps = 100  # Number of steps in the binomial tree
dividend_date = (datetime.strptime("04/11/2022", "%m/%d/%Y") - current_date_dt).days / 365.0  # Time to dividend date
dividend_amount = 0.88 # Discrete dividend amount

# Calculate the binomial tree prices for American call and put options with discrete dividends
bt_call_price = binomial_tree_american(True, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility, N_steps, dividend_date, dividend_amount)
bt_put_price = binomial_tree_american(False, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility, N_steps, dividend_date, dividend_amount)
print(bt_call_price)
# Calculate the binomial tree prices for American call and put options without discrete dividends
bt_call_price_no_div = binomial_tree_american(True, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility, N_steps, dividend_date, 0)
bt_put_price_no_div = binomial_tree_american(False, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility, N_steps, dividend_date, 0)


#Sensitivity of the call and put to a change in the dividend amount
# Define a small change in dividend amount for sensitivity analysis
dividend_change = 0.01  # Increase dividend by $0.01 for sensitivity

# Calculate new prices with increased dividend for call and put
bt_call_price_div_increase = binomial_tree_american(
    True, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, 
    implied_volatility, N_steps, dividend_date, dividend_amount + dividend_change)

bt_put_price_div_increase = binomial_tree_american(
    False, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, 
    implied_volatility, N_steps, dividend_date, dividend_amount + dividend_change)

# Sensitivity (delta) with respect to dividend amount for call and put options
call_div_sensitivity = (bt_call_price_div_increase - bt_call_price) / dividend_change
put_div_sensitivity = (bt_put_price_div_increase - bt_put_price) / dividend_change

def american_option_greeks(call, underlying, strike, ttm, rf, b, ivol, N, dividend_date, dividend_amount, epsilon=1e-5):
    """
    Calculate Greeks for American options using finite difference with binomial tree model.

    Parameters:
    call (bool): True for call option, False for put option
    underlying (float): Current stock price
    strike (float): Strike price
    ttm (float): Time to maturity in years
    rf (float): Risk-free rate (as a decimal)
    b (float): Cost of carry (continuously compounded dividend yield)
    ivol (float): Implied volatility (as a decimal)
    N (int): Number of time steps in the binomial tree
    dividend_date (float): Time to dividend date (in years)
    dividend_amount (float): Dividend amount paid on dividend_date
    epsilon (float): Small change for finite difference

    Returns:
    dict: Approximated Greeks - Delta, Gamma, Theta, Vega, Rho
    """
    # Base price
    price_base = binomial_tree_american(call, underlying, strike, ttm, rf, b, ivol, N, dividend_date, dividend_amount)

    # Delta approximation
    price_up_delta = binomial_tree_american(call, underlying + epsilon, strike, ttm, rf, b, ivol, N, dividend_date, dividend_amount)
    delta = (price_up_delta - price_base) / epsilon

    # Gamma approximation
    price_down_delta = binomial_tree_american(call, underlying - epsilon, strike, ttm, rf, b, ivol, N, dividend_date, dividend_amount)
    gamma = (price_up_delta - 2 * price_base + price_down_delta) / (epsilon ** 2)

    # Theta approximation (1 day decrement)
    price_up_theta = binomial_tree_american(call, underlying, strike, ttm - epsilon / 365, rf, b, ivol, N, dividend_date, dividend_amount)
    theta = (price_up_theta - price_base) / (-epsilon)  # Negative because time to expiration decreases

    # Vega approximation
    price_up_vega = binomial_tree_american(call, underlying, strike, ttm, rf, b, ivol + epsilon, N, dividend_date, dividend_amount)
    vega = (price_up_vega - price_base) / epsilon

    # Rho approximation
    price_up_rho = binomial_tree_american(call, underlying, strike, ttm, rf + epsilon, b, ivol, N, dividend_date, dividend_amount)
    rho = (price_up_rho - price_base) / epsilon

    # Carry Rho approximation
    price_up_carry_rho = binomial_tree_american(call, underlying, strike, ttm, rf, b + epsilon, ivol, N, dividend_date, dividend_amount)
    c_rho = (price_up_carry_rho - price_base) / epsilon
    
    return {
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta / 365,  # Theta per day
        "Vega": vega / 100,  # Vega per 1% change in volatility
        "Rho": rho / 100,  # Rho per 1% change in interest rate
        "Carry Rho": c_rho / 100  # Rho per 1% change in carry cost
    }

# Define the print function to display all values in a single table
def print_option_values_table(call_gbsm, put_gbsm, call_bt_with_div, put_bt_with_div, call_bt_no_div, put_bt_no_div,
                              call_div_sensitivity, put_div_sensitivity):
    """
    Prints a table of option values for GBSM and binomial tree methods with and without dividends,
    and shows sensitivity of call and put options to changes in dividend amount.

    Parameters:
    call_gbsm (float): Call option price using GBSM model.
    put_gbsm (float): Put option price using GBSM model.
    call_bt_with_div (float): Call option price using binomial tree with dividends.
    call_bt_no_div (float): Call option price using binomial tree without dividends.
    put_bt_with_div (float): Put option price using binomial tree with dividends.
    put_bt_no_div (float): Put option price using binomial tree without dividends.
    call_div_sensitivity (float): Sensitivity of call option to a change in dividend.
    put_div_sensitivity (float): Sensitivity of put option to a change in dividend.
    """
    
    # Create a DataFrame to store and display values in a table
    data = {
        "Option Type": ["Call", "Put"],
        "GBSM": [call_gbsm, put_gbsm],
        "BT with Div.": [call_bt_with_div, put_bt_with_div],
        "BT without Div.": [call_bt_no_div, put_bt_no_div],
        "Div. Sensitivity": [call_div_sensitivity, put_div_sensitivity]
    }
    df = pd.DataFrame(data)
    
    # Display the table
    print()
    print("American Option Pricing and Sensitivity to Dividend Changes:")
    print(df)

# Call the function to print
print_option_values_table(call_gbsm, put_gbsm, bt_call_price, bt_put_price, bt_call_price_no_div, bt_put_price_no_div,
                          call_div_sensitivity, put_div_sensitivity)

# Calculate American option Greeks for both call and put options
american_call_greeks = american_option_greeks(True, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility, N_steps, dividend_date, dividend_amount)
american_put_greeks = american_option_greeks(False, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility, N_steps, dividend_date, dividend_amount)

american_call_greeks_no_div = american_option_greeks(True, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility, N_steps, dividend_date, 0)
american_put_greeks_no_div = american_option_greeks(False, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, implied_volatility, N_steps, dividend_date, 0)

# Function to print Greeks for both Call and Put options with and without dividends
def print_greeks_table(call_greeks, put_greeks, call_greeks_no_div, put_greeks_no_div):
    """
    Prints the option Greeks for Call and Put options with and without dividends in a single table.

    Parameters:
    call_greeks (dict): Greeks for Call option with dividends.
    put_greeks (dict): Greeks for Put option with dividends.
    call_greeks_no_div (dict): Greeks for Call option without dividends.
    put_greeks_no_div (dict): Greeks for Put option without dividends.
    """
    # Prepare data for the table
    data = {
        "Greek": list(call_greeks.keys()),
        "Call with Dividend": list(call_greeks.values()),
        "Put with Dividend": list(put_greeks.values()),
        "Call without Dividend": list(call_greeks_no_div.values()),
        "Put without Dividend": list(put_greeks_no_div.values())
    }
    
    # Create a DataFrame to display in table format
    df = pd.DataFrame(data)
    
    # Display the table
    print("\nAmerican Option Greeks Comparison Table:")
    print(df.to_string(index=False))

print_greeks_table(american_call_greeks, american_put_greeks, american_call_greeks_no_div, american_put_greeks_no_div)
