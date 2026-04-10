def user_based_cf_prediction(similarities, ratings):
    """
    Predict a rating using user-based collaborative filtering.
    """
    weighted_sum = 0.0
    sim_sum = 0.0

    for sim, rating in zip(similarities, ratings):
        if sim > 0:
            weighted_sum += sim * rating
            sim_sum += sim

    if sim_sum == 0:
        return 0.0

    return weighted_sum / sim_sum