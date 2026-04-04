import math

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    epsilon = 1e-9
    n = len(X_train)
    d = len(X_train[0])

    classes = sorted(set(y_train))
    stats = {}

    for c in classes:
        X_c = [X_train[i] for i in range(n) if y_train[i] == c]
        n_c = len(X_c)

        prior = n_c / n
        means = []
        variances = []

        for j in range(d):
            vals = [row[j] for row in X_c]
            mean = sum(vals) / n_c
            var = sum((x - mean) ** 2 for x in vals) / n_c
            means.append(mean)
            variances.append(var + epsilon)

        stats[c] = (prior, means, variances)

    predictions = []

    for x in X_test:
        best_class = None
        best_log_prob = -float("inf")

        for c in classes:
            prior, means, variances = stats[c]
            log_prob = math.log(prior)

            for j in range(d):
                mean = means[j]
                var = variances[j]
                log_prob += -0.5 * math.log(2 * math.pi * var) - ((x[j] - mean) ** 2) / (2 * var)

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = c

        predictions.append(best_class)

    return predictions