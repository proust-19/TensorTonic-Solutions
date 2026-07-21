def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    if not isinstance(relevant, set):
        relevant = set(relevant)

    if k <= 0 or len(relevant) == 0:
        return [0.0, 0.0]

    top_k = recommended[:min(k, len(recommended))]
    rel_k = sum(1 for top in top_k if top in relevant)

    precision = rel_k / len(top_k) if len(top_k) > 0 else 0.0
    recall = rel_k / len(relevant)
    
    return [precision, recall]