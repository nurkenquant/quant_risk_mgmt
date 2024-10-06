import numpy as np
import pandas as pd

# Given parameters
sigma_squared = 0.01  # Variance of r_t
sigma = np.sqrt(sigma_squared)
P_t_minus_1 = 100  # this is an assumption that P_{t-1} = 100
n_simulations = 1000  # Number of simulations

# Generate r_t ~ N(0, 0.01)
r_t = np.random.normal(0, sigma, n_simulations)

# Classical Brownian Motion: P_t = P_{t-1} + r_t
P_t_classical = P_t_minus_1 + r_t

# Arithmetic Return System: P_t = P_{t-1} * (1 + r_t)
P_t_arithmetic = P_t_minus_1 * (1 + r_t)

# Log Return System (Geometric Brownian Motion): P_t = P_{t-1} * e^(r_t)
P_t_log = P_t_minus_1 * np.exp(r_t)

# Calculate mean and standard deviation for each return system
mean_classical = np.mean(P_t_classical)
std_classical = np.std(P_t_classical)

mean_arithmetic = np.mean(P_t_arithmetic)
std_arithmetic = np.std(P_t_arithmetic)

mean_log = np.mean(P_t_log)
std_log = np.std(P_t_log)

# Display results
results = {
    "Return Type": ["Classical Brownian", "Arithmetic Return", "Log Return"],
    "Expected Value (Mean)": [mean_classical, mean_arithmetic, mean_log],
    "Standard Deviation": [std_classical, std_arithmetic, std_log]
}

results_df = pd.DataFrame(results)
print(results_df)