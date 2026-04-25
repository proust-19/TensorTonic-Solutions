def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    T = len(rewards)
    G = [0.0] * T
    rr = 0.0

    for t in reversed(range(T)):
        rr = rewards[t] + gamma * rr
        G[t] = rr

    mean_G = sum(G)/T
    adv = [g - mean_G for g in G]
    
    loss = -sum(lp*a for lp, a in zip(log_probs, adv))/T
    return loss