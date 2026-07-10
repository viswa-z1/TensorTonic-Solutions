def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    result = []

    for req in requests:
        offline = feature_store.get(req["user_id"], defaults)
        combined = {**offline, **req["online_features"]}
        result.append(combined)

    return result