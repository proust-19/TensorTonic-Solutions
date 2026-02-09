def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X = np.array(X, "float")
    y = np.array(y, "float")

    l, d = X.shape
    I = [[lam if i == j else 0 for j in range(d)]for i in range(d)]
        
    inn = (np.transpose(X) @ X) + I
    w = np.linalg.inv(inn) @ np.transpose(X) @ y

    return w.tolist()