def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    best = max(
        models,
        key=lambda m: (
            m["accuracy"],      # Higher accuracy
            -m["latency"],      # Lower latency
            m["timestamp"]      # More recent date (YYYY-MM-DD)
        )
    )
    return best["name"]