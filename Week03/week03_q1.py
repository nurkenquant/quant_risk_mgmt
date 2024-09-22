import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the DailyReturn.csv file
file_path = "DailyReturn.csv"
daily_returns = pd.read_csv(file_path)

# Drop the 'SPY' column for stock data only
stock_returns = daily_returns.drop(columns=['SPY']).values

# Define a function to calculate exponentially weighted covariance using pandas
def pandas_ewm_covariance(data, lambda_val):
    # Convert lambda to span parameter for pandas ewm
    span = 1 / (1 - lambda_val)
    
    # Calculate exponentially weighted covariance matrix
    ewm_cov = data.ewm(span=span).cov(pairwise=True)
    
    # Get the last available covariance matrix from the EWM
    last_date = ewm_cov.index.get_level_values(0).max()
    last_cov = ewm_cov.loc[last_date]
    
    return last_cov


# Define a function to calculate exponentially weighted covariance manually
def manually_ewm_covariance(data, lambda_val):
    # Calculate the number of observations
    num_observations = data.shape[0]
    
    # Initialize weights for exponentially weighted moving calculation
    weights = np.array([(1 - lambda_val) * (lambda_val ** i) for i in range(num_observations)][::-1])
    
    # Normalize weights so that they sum to 1
    weights /= weights.sum()
    
    # Calculate weighted mean
    weighted_mean = np.dot(weights, data)
    
    # Calculate weighted deviations
    demeaned_data = data - weighted_mean
    
    # Calculate the exponentially weighted covariance matrix
    weighted_cov_matrix = np.dot(demeaned_data.T * weights, demeaned_data)
    
    return weighted_cov_matrix

# Ensure stock_returns is a pandas DataFrame, not a numpy array
stock_returns_df = pd.DataFrame(stock_returns, columns=daily_returns.columns[1:])


# Choose a lambda value for exponential weighting
lambda_value = 0.97

# Calculate the exponentially weighted covariance matrix using pandas
ewm_cov_matrix = pandas_ewm_covariance(stock_returns_df, lambda_value)
# Calculate the manually implemented exponentially weighted covariance matrix
manual_cov_matrix = manually_ewm_covariance(stock_returns_df, lambda_value)


# Save the pandas implemented exponentially weighted covariance matrix to an Excel file
pandas_ewm_cov_file_xlsx = "pandas_ewm_cov_matrix.xlsx"
ewm_cov_matrix.to_excel(pandas_ewm_cov_file_xlsx, index=True)

# Convert the result to a DataFrame with appropriate headers
manual_cov_df = pd.DataFrame(
    manual_cov_matrix, 
    index=daily_returns.columns[1:], 
    columns=daily_returns.columns[1:]
)

# Save the manually implemented exponentially weighted covariance matrix to an Excel file
manual_cov_file_xlsx = "manual_ewm_cov_matrix.xlsx"
manual_cov_df.to_excel(manual_cov_file_xlsx, index=True)

# Calculate the difference between the two covariance matrices
cov_difference = manual_cov_df - ewm_cov_matrix

# Save the difference to an Excel file
difference_file_xlsx = "difference_ewm_cov_matrix.xlsx"
cov_difference.to_excel(difference_file_xlsx, index=True)

# Function to perform PCA and plot cumulative variance explained
def plot_pca_variance(cov_matrix, lambda_val, title, color):
    # Perform PCA on the covariance matrix
    pca = PCA()
    pca.fit(cov_matrix)
    
    # Calculate cumulative variance explained
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Plot the cumulative variance explained
    plt.plot(cumulative_variance, marker='o', color=color, label=title)

# Vary lambda from a range of values and plot the PCA cumulative variance for each
lambda_values = [0.8, 0.9, 0.95, 0.97, 0.99]

for lam in lambda_values:
    # Calculate the exponentially weighted covariance matrix for the current lambda manually and using pandas
    manual_ewm_cov_matrix = manually_ewm_covariance(stock_returns_df.values, lam)
    pandas_ewm_cov_matrix = pandas_ewm_covariance(stock_returns_df, lam)
    
    # Plot the PCA cumulative variance explained for manual and pandas implementations
    plt.figure(figsize=(10, 6))
    plot_pca_variance(manual_ewm_cov_matrix, lam, f'Manual EWM Covariance (λ={lam})', 'blue')
    plot_pca_variance(pandas_ewm_cov_matrix, lam, f'Pandas EWM Covariance (λ={lam})', 'orange')
    
    # Add labels and legend
    plt.title(f'Cumulative Variance Explained by PCA for Lambda = {lam}')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.legend()
    plt.grid(True)
    plt.show()