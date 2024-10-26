from risk_management_lib.risk_management.risk_measures.var_calculations import var_normal, var_historical, var_t
from risk_management_lib.risk_management.risk_measures.es_calculations import es_normal, es_historical, es_t
from risk_management_lib.risk_management.utils.helper_functions import fit_generalized_t

import pandas as pd

# Load the data from 'problem1.csv'
data_path = "Week05/problem1.csv"
data = pd.read_csv(data_path)

# Extract the returns data
returns = data['x'].values

mu, sigma, nu = fit_generalized_t(returns)
# Assuming returns is a numpy array of historical returns
var_ewma = var_normal(returns, 0.99, lambda_value=0.97)
var_hist = var_historical(returns, 0.99)
var_tdist = var_t(mu, sigma, nu, 0.99)

es_ewma = es_normal(returns, 0.99, lambda_value=0.97)
es_hist = es_historical(returns, 0.99)
es_tdist = es_t(mu, sigma, nu, 0.99)

print("EWMA VaR:", var_ewma, "EWMA ES:", es_ewma)
print("Historical VaR:", var_hist, "Historical ES:", es_hist)
print("T-distribution VaR:", var_tdist, "T-distribution ES:", es_tdist)
