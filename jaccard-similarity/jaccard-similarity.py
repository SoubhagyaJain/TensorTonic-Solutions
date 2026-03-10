def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    a = set(set_a)
    b = set(set_b)

    union = a | b
    if not union:   # both sets are empty
        return 0.0

    intersection = a & b
    return len(intersection) / len(union)