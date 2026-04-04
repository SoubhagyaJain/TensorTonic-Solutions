import math

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    n = len(actual_tokens)
    cross_entropy = 0.0

    for i in range(n):
        p = prob_distributions[i][actual_tokens[i]]
        cross_entropy += -math.log(p)

    cross_entropy /= n
    return math.exp(cross_entropy)