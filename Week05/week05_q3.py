import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from risk_management_lib.risk_management.utils.helper_functions import return_calculate, fit_generalized_t, ew_variance
from risk_management_lib.risk_management.risk_measures.var_calculations import var_t, var_normal
from risk_management_lib.risk_management.risk_measures.es_calculations import es_t, es_normal
from risk_management_lib.risk_management.simulations.copula_simulation import copula_based_var_es

# Load data
portfolio_path = "portfolio.csv"
portfolio = pd.read_csv(portfolio_path)
daily_prices_path = "DailyPrices.csv"
prices = pd.read_csv(daily_prices_path)

# parameters
confidence_level = 0.99
lambda_value = 0.97

# Calculate arithmetic returns from daily prices
returns = return_calculate(prices, method='arithmetic')

# Separate stocks based on portfolios
portfolio_a = portfolio[portfolio['Portfolio'] == 'A']
portfolio_b = portfolio[portfolio['Portfolio'] == 'B']
portfolio_c = portfolio[portfolio['Portfolio'] == 'C']

# Calculate VaR and ES for Portfolio A and B using Generalized T-distribution
VaRs_A, ESs_A = [], []
for stock in portfolio_a['Stock']:
    if stock in returns.columns:
        # Fit generalized T-distribution for each stock in Portfolio A
        mu, sigma, nu = fit_generalized_t(returns[stock])
        VaRs_A.append(var_t(mu, sigma, nu, confidence_level))
        ESs_A.append(es_t(mu, sigma, nu, confidence_level))

VaRs_B, ESs_B = [], []
for stock in portfolio_b['Stock']:
    if stock in returns.columns:
        # Fit generalized T-distribution for each stock in Portfolio B
        mu, sigma, nu = fit_generalized_t(returns[stock])
        VaRs_B.append(var_t(mu, sigma, nu, confidence_level))
        ESs_B.append(es_t(mu, sigma, nu, confidence_level))

# Calculate VaR and ES for Portfolio C using Normal distribution
VaRs_C, ESs_C = [], []
for stock in portfolio_c['Stock']:
    if stock in returns.columns:
        mu = returns[stock].mean()
        sigma = returns[stock].std()
        VaRs_C.append(var_normal(returns[stock], confidence_level, lambda_value))
        ESs_C.append(es_normal(returns[stock], confidence_level, lambda_value))

# Aggregate VaR and ES for each portfolio
VaR_A, ES_A = sum(VaRs_A), sum(ESs_A)
VaR_B, ES_B = sum(VaRs_B), sum(ESs_B)
VaR_C, ES_C = sum(VaRs_C), sum(ESs_C)

# Step 3: Calculate Combined VaR and ES using Copula
def calculate_portfolio_returns(portfolio, returns_df, prices_df):
    """
    Aggregate returns for each portfolio based on the stock holdings and current asset values.
    """
    latest_prices = prices_df.iloc[-1]
    portfolio_returns = 0
    total_value = 0
    
    for index, row in portfolio.iterrows():
        stock = row['Stock']
        holding = row['Holding']
        if stock in returns_df.columns and stock in latest_prices.index:
            asset_value = holding * latest_prices[stock]
            portfolio_returns += (asset_value * returns_df[stock])
            total_value += asset_value

    return portfolio_returns / total_value if total_value != 0 else portfolio_returns

portfolio_agg = pd.DataFrame({
    'Portfolio_A': calculate_portfolio_returns(portfolio_a, returns, prices),
    'Portfolio_B': calculate_portfolio_returns(portfolio_b, returns, prices),
    'Portfolio_C': calculate_portfolio_returns(portfolio_c, returns, prices)
})

correlation_matrix = portfolio_agg.corr().values
portfolio_vars = [VaR_A, VaR_B, VaR_C]
portfolio_ess = [ES_A, ES_B, ES_C]
combined_VaR, combined_ES = copula_based_var_es(portfolio_vars, portfolio_ess, correlation_matrix)

# Comparison with EWMA VaR from Week 4
def calculate_portfolio_ewma_var(portfolio, returns_df, prices_df, lambda_value, confidence_level):
    """
    Calculate EWMA-based VaR and ES for a given portfolio.

    Parameters:
        portfolio (DataFrame): Portfolio DataFrame with 'Stock' and 'Holding'.
        returns_df (DataFrame): DataFrame of stock returns.
        prices_df (DataFrame): DataFrame of daily prices.
        lambda_value (float): Decay factor for EWMA.
        confidence_level (float): Confidence level for VaR and ES calculation.

    Returns:
        tuple: (VaR, ES) calculated using EWMA.
    """
    # Get the portfolio returns time series
    portfolio_returns = calculate_portfolio_returns(portfolio, returns_df, prices_df)
    
    # Calculate VaR and ES using EWMA from portfolio returns
    VaR_EWMA = var_normal(portfolio_returns, confidence_level, lambda_value)
    ES_EWMA = es_normal(portfolio_returns, confidence_level, lambda_value)
    
    return VaR_EWMA, ES_EWMA

# Calculate EWMA-based VaR and ES for each portfolio
VaR_EWMA_A, ES_EWMA_A = calculate_portfolio_ewma_var(portfolio_a, returns, prices, lambda_value, confidence_level)
VaR_EWMA_B, ES_EWMA_B = calculate_portfolio_ewma_var(portfolio_b, returns, prices, lambda_value, confidence_level)
VaR_EWMA_C, ES_EWMA_C = calculate_portfolio_ewma_var(portfolio_c, returns, prices, lambda_value, confidence_level)

# Sum Individual Portfolio VaRs for Total EWMA VaR and ES
total_ewma_var = VaR_EWMA_A + VaR_EWMA_B + VaR_EWMA_C
total_ewma_es = ES_EWMA_A + ES_EWMA_B + ES_EWMA_C

# Output
print("Norm distr with an EWMA method: ")
print(f"VaR (Portfolio A): {abs(VaR_EWMA_A):.4f}")
print(f"ES (Portfolio A): {ES_EWMA_A:.4f}")

print(f"VaR (Portfolio B): {abs(VaR_EWMA_B):.4f}")
print(f"ES (Portfolio B): {ES_EWMA_B:.4f}")

print(f"VaR (Portfolio C): {abs(VaR_EWMA_C):.4f}")
print(f"ES (Portfolio C): {ES_EWMA_C:.4f}")

print(f"VaR (Total of Portfolio A, B, and C): {abs(total_ewma_var):.4f}")
print(f"ES (Total of Portfolio A, B, and C): {total_ewma_es:.4f}")

print("\nCombined Approach (Copula and Distributions): ")
print(f"VaR (Portfolio A, T-Distr.): {abs(VaR_A):.4f}")
print(f"ES (Portfolio A, T-Distr.): {ES_A:.4f}")

print(f"VaR (Portfolio B, T-Distr.): {abs(VaR_B):.4f}")
print(f"ES (Portfolio B, T-Distr.): {ES_B:.4f}")

print(f"VaR (Portfolio C, Normal distr.): {abs(VaR_C):.4f}")
print(f"ES (Portfolio C, Normal distr.): {ES_C:.4f}")

print(f"VaR (Combined Portfolio, Copula-Based): {abs(combined_VaR):.4f}")
print(f"ES (Combined Portfolio, Copula-Based): {combined_ES:.4f}")
