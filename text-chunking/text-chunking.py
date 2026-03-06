def text_chunking(tokens, chunk_size, overlap):
    step = chunk_size - overlap
    result = []

    start = 0
    n = len(tokens)

    while start < n:
        result.append(tokens[start:start + chunk_size])

        if start + chunk_size >= n:
            break

        start += step

    return result