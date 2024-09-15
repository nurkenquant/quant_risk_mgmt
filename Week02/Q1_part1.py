import numpy as np
import pandas as pd

# Load the dataset
file_path = 'problem1.csv'
data = pd.read_csv(file_path)

# Extract the column values
x = data['x']

# Number of samples
n = len(x)

# First moment (mean)
mean = np.sum(x) / n

# Second moment (variance) - normalized by n, biased
variance = np.sum((x - mean) ** 2) / n

# Third moment (skewness) - normalized by sigma^3
skewness = np.sum((x - mean) ** 3) / (n * (np.sqrt(variance) ** 3))

# Fourth moment (kurtosis) - normalized by sigma^4, and subtract 3 for excess kurtosis
kurtosis = np.sum((x - mean) ** 4) / (n * (variance ** 2)) - 3

# Display the calculated moments
print("Mean:", mean)
print("Variance:", variance)
print("Skewness:", skewness)
print("Kurtosis (Excess):", kurtosis)
