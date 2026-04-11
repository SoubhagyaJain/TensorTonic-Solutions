def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    Returns the output as a 2D list of floats.
    """
    h, w = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])

    # Pad image with zeros
    padded_h = h + 2 * padding
    padded_w = w + 2 * padding
    padded = [[0] * padded_w for _ in range(padded_h)]

    for i in range(h):
        for j in range(w):
            padded[i + padding][j + padding] = image[i][j]

    # Output dimensions
    out_h = ((h + 2 * padding - kh) // stride) + 1
    out_w = ((w + 2 * padding - kw) // stride) + 1

    output = []

    for i in range(out_h):
        row = []
        for j in range(out_w):
            total = 0.0
            for m in range(kh):
                for n in range(kw):
                    total += padded[i * stride + m][j * stride + n] * kernel[m][n]
            row.append(total)
        output.append(row)

    return output