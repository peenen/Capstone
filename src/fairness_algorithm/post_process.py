def apply_reranking(predictions, config):
    # simple random perturbation to promote fairness
    return predictions.sample(frac=1.0, random_state=42).reset_index(drop=True)