import numpy as np

def feed_forward(x, W1, b1, W2, b2):
    # First linear layer
    hidden = np.dot(x, W1) + b1

    # ReLU activation
    hidden = np.maximum(0, hidden)

    # Second linear layer
    output = np.dot(hidden, W2) + b2

    return output