def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k = recommended[:k]

    to = 0
    for top in top_k:
        if top in relevant:
            to += 1

    return [to/k, to/len(relevant)]