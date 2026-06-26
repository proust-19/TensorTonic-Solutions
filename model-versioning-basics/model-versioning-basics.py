def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    if not models:
        return None
        
    best = max(
        models,
        key=lambda m: (m["accuracy"], -m["latency"], m["timestamp"])
    )

    return best["name"]