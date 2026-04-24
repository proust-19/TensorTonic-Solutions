def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    def f(x):
        return 2*a*x + b
        
    for i in range(steps):
        x -= lr*f(x)
    return x    