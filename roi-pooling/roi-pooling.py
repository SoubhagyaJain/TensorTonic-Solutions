import numpy as np

def roi_pool(feature_map, rois, output_size):
    fm = np.asarray(feature_map)
    rois = np.asarray(rois, dtype=np.int64)

    H, W = fm.shape
    R = rois.shape[0]

    out = np.empty((R, output_size, output_size), dtype=fm.dtype)

    for r in range(R):
        x1, y1, x2, y2 = rois[r]
        roi_w = x2 - x1
        roi_h = y2 - y1

        for i in range(output_size):
            h_start = y1 + (i * roi_h) // output_size
            h_end   = y1 + ((i + 1) * roi_h) // output_size
            if h_end == h_start:
                h_end = h_start + 1

            hs = max(0, min(H, h_start))
            he = max(0, min(H, h_end))
            if he <= hs:
                he = min(H, hs + 1)

            for j in range(output_size):
                w_start = x1 + (j * roi_w) // output_size
                w_end   = x1 + ((j + 1) * roi_w) // output_size
                if w_end == w_start:
                    w_end = w_start + 1

                ws = max(0, min(W, w_start))
                we = max(0, min(W, w_end))
                if we <= ws:
                    we = min(W, ws + 1)

                out[r, i, j] = np.max(fm[hs:he, ws:we])

    # ✅ must return a list of 2D lists
    return out.tolist()