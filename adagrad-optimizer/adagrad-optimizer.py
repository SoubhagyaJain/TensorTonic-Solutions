import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    G = np.array(G, dtype=float)

    new_G = G + g**2
    new_w = w - (lr / np.sqrt(new_G + eps)) * g

    return new_w.tolist(), new_G.tolist()
    