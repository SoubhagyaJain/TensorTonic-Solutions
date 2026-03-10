def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    rows = len(data)
    cols = len(data[0])

    # Find min and max for each column
    col_mins = [float("inf")] * cols
    col_maxs = [float("-inf")] * cols

    for row in data:
        for j in range(cols):
            if row[j] < col_mins[j]:
                col_mins[j] = row[j]
            if row[j] > col_maxs[j]:
                col_maxs[j] = row[j]

    # Scale each value
    result = []
    for row in data:
        scaled_row = []
        for j in range(cols):
            mn = col_mins[j]
            mx = col_maxs[j]
            if mx == mn:
                scaled_row.append(0.0)
            else:
                scaled_row.append((row[j] - mn) / (mx - mn))
        result.append(scaled_row)

    return result