def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    x = np.array(X)
    w = np.array(W)
    b = np.array(b)
    
    y = np.matmul(X, W) + b

    return y.tolist()