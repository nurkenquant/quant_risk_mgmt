# Monte Carlo simulation methods
import numpy as np

def monte_carlo_simulation(initial_value, num_simulations, num_steps, mu, sigma):
    paths = np.zeros((num_steps, num_simulations))
    paths[0] = initial_value
    for t in range(1, num_steps):
        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) + sigma * np.random.normal(size=num_simulations))
    return paths
