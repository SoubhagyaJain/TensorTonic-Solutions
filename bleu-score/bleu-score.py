import math

def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    if not candidate:
        return 0.0

    c = len(candidate)
    r = len(reference)
    precisions = []

    for n in range(1, max_n + 1):
        cand_total = c - n + 1
        if cand_total <= 0:
            return 0.0

        cand_counts = {}
        ref_counts = {}

        for i in range(cand_total):
            ng = tuple(candidate[i:i + n])
            cand_counts[ng] = cand_counts.get(ng, 0) + 1

        for i in range(r - n + 1):
            ng = tuple(reference[i:i + n])
            ref_counts[ng] = ref_counts.get(ng, 0) + 1

        clipped = 0
        for ng, count in cand_counts.items():
            clipped += min(count, ref_counts.get(ng, 0))

        p_n = clipped / cand_total
        if p_n == 0:
            return 0.0
        precisions.append(p_n)

    if c >= r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c)

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    return bp * geo_mean