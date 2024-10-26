# Copula simulation methods
import numpy as np
from scipy.stats import norm, t, multivariate_normal

def copula_based_var_es(portfolio_vars, portfolio_ess, correlation_matrix, confidence_level=0.99):
    """
    Aggregate VaR and ES of multiple portfolios using a copula.

    Parameters:
        portfolio_vars (list): List of individual portfolio VaR values (e.g., [var_a, var_b, var_c]).
        portfolio_ess (list): List of individual portfolio ES values (e.g., [es_a, es_b, es_c]).
        correlation_matrix (np.array): Correlation matrix of the portfolios (3x3).
        confidence_level (float): Confidence level for VaR and ES.

    Returns:
        tuple: Combined VaR and ES values.
    """
    # if len(portfolio_vars) != 3 or len(portfolio_ess) != 3 or correlation_matrix.shape != (3, 3):
    #    raise ValueError("The function requires 3 portfolios and a 3x3 correlation matrix.")

    # Generate correlated samples using the copula
    copula = multivariate_normal(mean=[0, 0, 0], cov=correlation_matrix)
    samples = copula.rvs(size=10000)

    # Transform to uniform via CDF
    uniforms = norm.cdf(samples)

    # Transform back to marginal distributions via quantiles
    combined_losses = []
    for i in range(10000):
        loss = (t.ppf(uniforms[i, 0], df=5) * portfolio_vars[0] +
                t.ppf(uniforms[i, 1], df=5) * portfolio_vars[1] +
                norm.ppf(uniforms[i, 2]) * portfolio_vars[2])
        combined_losses.append(loss)

    # Calculate combined VaR and ES
    combined_losses = np.array(combined_losses)
    combined_var = np.percentile(combined_losses, 100 * (1 - confidence_level))
    combined_es = np.mean(combined_losses[combined_losses <= combined_var])
    
    return abs(combined_var), abs(combined_es)
