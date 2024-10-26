# Bootstrap simulation methods
import numpy as np

def bootstrap_simulation(returns, num_samples):
    return np.random.choice(returns, size=num_samples, replace=True)
