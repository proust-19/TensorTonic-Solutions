import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    x = np.asarray(x)
    n = len(x)
    
    me = np.mean(x)
    s = np.std(x, ddof = 1)

    t_stat = (me - mu0) / (s / np.sqrt(n))

    return t_stat