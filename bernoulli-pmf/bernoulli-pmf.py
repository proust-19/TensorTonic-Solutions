import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    x = np.asarray(x)
    pmf = np.where(x == 1, p, 1-p)
    
    mean = p
    var = p*(1-p)
    return np.asarray(pmf), mean, var