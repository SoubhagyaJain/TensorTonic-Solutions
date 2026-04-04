def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    result = values[:]
    n = len(result)
    i = 0

    while i < n:
        if result[i] is None:
            left = i - 1
            j = i
            while j < n and result[j] is None:
                j += 1
            right = j

            left_val = result[left]
            right_val = result[right]
            gap = right - left

            for k in range(1, gap):
                result[left + k] = left_val + (k / gap) * (right_val - left_val)

            i = right
        else:
            i += 1

    return result