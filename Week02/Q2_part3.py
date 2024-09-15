import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load the dataset
file_path = 'problem2_x.csv'
data = pd.read_csv(file_path)

# Estimate the means and covariance matrix
mean_x1 = np.mean(data['x1'])
mean_x2 = np.mean(data['x2'])
cov_matrix = np.cov(data['x1'], data['x2'])

# Extract covariance terms
cov_x1x1 = cov_matrix[0, 0]
cov_x2x2 = cov_matrix[1, 1]
cov_x1x2 = cov_matrix[0, 1]

# Calculate the conditional mean and variance for X2 given X1
def conditional_mean_variance(x1):
    cond_mean_x2 = mean_x2 + (cov_x1x2 / cov_x1x1) * (x1 - mean_x1)
    cond_var_x2 = cov_x2x2 - (cov_x1x2**2 / cov_x1x1)
    return cond_mean_x2, cond_var_x2

# Calculate conditional means, variances, and confidence intervals
x1_values = data['x1']
cond_means = []
cond_intervals_lower = []
cond_intervals_upper = []

for x1 in x1_values:
    cond_mean, cond_var = conditional_mean_variance(x1)
    cond_means.append(cond_mean)
    conf_interval = norm.interval(0.95, loc=cond_mean, scale=np.sqrt(cond_var))
    cond_intervals_lower.append(conf_interval[0])
    cond_intervals_upper.append(conf_interval[1])

# Convert lists to numpy arrays for compatibility
cond_means = np.array(cond_means)
cond_intervals_lower = np.array(cond_intervals_lower)
cond_intervals_upper = np.array(cond_intervals_upper)
x1_values = np.array(data['x1'])

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x1_values, data['x2'], label='Observed X2', color='blue')
plt.plot(x1_values, cond_means, label='Expected X2', color='red', linestyle='--')
plt.fill_between(x1_values, cond_intervals_lower, cond_intervals_upper, color='gray', alpha=0.3, label='95% CI')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Conditional Expectation and 95% Confidence Interval for X2 given X1')
plt.legend()
plt.grid(True)
plt.show()
