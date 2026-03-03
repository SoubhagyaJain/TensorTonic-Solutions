import numpy as np
from collections import Counter

def tfidf_vectorizer(documents):
    """
    TF-IDF vectorizer.
    Accepts: list[str] OR np.ndarray of strings
    Returns: (tfidf_matrix, vocabulary)
      - tfidf_matrix: (n_docs, n_vocab) float64
      - vocabulary: list[str] sorted alphabetically
    """
    if documents is None:
        docs = []
    else:
        # Accept numpy arrays and other array-likes
        docs = np.asarray(documents, dtype=object).ravel().tolist()

    N = len(docs)
    if N == 0:
        return np.zeros((0, 0), dtype=np.float64), []

    doc_counters = []
    doc_lengths = np.zeros(N, dtype=np.int64)
    df_counts = {}  # term -> document frequency

    # 1) Tokenize + count TF per doc and DF across docs
    for i, doc in enumerate(docs):
        if doc is None:
            text = ""
        else:
            text = str(doc)

        text = text.strip()
        if not text:
            doc_counters.append(Counter())
            doc_lengths[i] = 0
            continue

        tokens = text.lower().split()
        doc_lengths[i] = len(tokens)

        c = Counter(tokens)
        doc_counters.append(c)

        for term in c.keys():
            df_counts[term] = df_counts.get(term, 0) + 1

    # 2) Vocabulary sorted alphabetically
    vocabulary = sorted(df_counts.keys())
    V = len(vocabulary)
    if V == 0:
        return np.zeros((N, 0), dtype=np.float64), []

    term2idx = {t: j for j, t in enumerate(vocabulary)}

    # 3) IDF = log(N / df)
    df_vec = np.fromiter((df_counts[t] for t in vocabulary), dtype=np.float64, count=V)
    idf = np.log(N / df_vec)

    # 4) Dense TF-IDF matrix
    tfidf = np.zeros((N, V), dtype=np.float64)
    for i, c in enumerate(doc_counters):
        total = doc_lengths[i]
        if total == 0:
            continue
        inv_total = 1.0 / float(total)
        for term, cnt in c.items():
            j = term2idx[term]
            tfidf[i, j] = (cnt * inv_total) * idf[j]

    return tfidf, vocabulary