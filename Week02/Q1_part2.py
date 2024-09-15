import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# Load the dataset
file_path = 'problem1.csv'
data = pd.read_csv(file_path)

# Extract the column values
x = data['x']

# First moment (mean) using pandas
mean_by_pkg = x.mean()

# Second moment (variance) using pandas
variance_by_pkg = x.var() # by default unbiased variance using 𝑛−1 in the denominator (ddof=1)

# Third moment (skewness) using scipy.stats
skewness_by_pkg = skew(x) # by default bias=True

# Fourth moment (kurtosis) using scipy.stats (excess kurtosis, subtracting 3)
kurtosis_by_pkg = kurtosis(x) # by default bias=True

# Display the results from the statistical package
print("Mean (package):", mean_by_pkg)
print("Variance (package):", variance_by_pkg)
print("Skewness (package):", skewness_by_pkg)
print("Kurtosis (Excess, package):", kurtosis_by_pkg)
