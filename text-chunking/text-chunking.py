def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with overlap.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(tokens), step):
        chunk = tokens[start:start + chunk_size]
        chunks.append(chunk)

        if start + chunk_size >= len(tokens):
            break

    return chunks