def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """
    # Write code here
import math

def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """
    
    # Return 0.0 if recommendation list is empty
    if not recommendations:
        return 0.0
    
    total_novelty = 0
    
    for item in recommendations:
        popularity = item_counts[item] / n_users
        total_novelty += -math.log2(popularity)
    
    return total_novelty / len(recommendations)