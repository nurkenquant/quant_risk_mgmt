import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load the data from the provided CSV file
file_path = 'problem3.csv'
data = pd.read_csv(file_path)

# Plot the data to visualize its structure
plt.figure(figsize=(10, 6))
plt.plot(data, label='Time Series Data')
plt.title('Time Series Data')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()

# Plot the ACF and PACF to identify patterns in autocorrelation
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

plot_acf(data, ax=ax[0], lags=20)
ax[0].set_title('Autocorrelation Function (ACF)')

plot_pacf(data, ax=ax[1], lags=20)
ax[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

# Initialize a dictionary to store model results
results = {'Model': [], 'AIC': [], 'BIC': []}

# Fit AR(1), AR(2), AR(3) models
for p in range(1, 4):
    model = ARIMA(data['x'], order=(p, 0, 0))  # AR(p)
    model_fit = model.fit()
    results['Model'].append(f'AR({p})')
    results['AIC'].append(model_fit.aic)
    results['BIC'].append(model_fit.bic)

# Fit MA(1), MA(2), MA(3) models
for q in range(1, 4):
    model = ARIMA(data['x'], order=(0, 0, q))  # MA(q)
    model_fit = model.fit()
    results['Model'].append(f'MA({q})')
    results['AIC'].append(model_fit.aic)
    results['BIC'].append(model_fit.bic)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Sort results by AIC in ascending order
sorted_results_df = results_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

# Display sorted results in a standard print format
print("Sorted ARIMA Model Results:")
print(sorted_results_df[['Model', 'AIC', 'BIC']])