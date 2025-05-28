def evaluate_f1_at_k(predictions, ground_truth, k=10):
    """
    Compute F1@K for precision-recall balance at top-K recommendations.
    Inputs:
        predictions: list of recommended item IDs per user (dict or DataFrame)
        ground_truth: list of true item IDs per user
    Returns:
        float: F1 score at top-K
    """
    # TODO: Implement actual F1@K computation
    return 0.0

def evaluate_ndcg_at_k(predictions, ground_truth, k=10):
    """
    Compute Normalized Discounted Cumulative Gain at top-K
    """
    # TODO: Implement NDCG@K
    return 0.0

def evaluate_auc(predictions, ground_truth):
    """
    Evaluate Area Under Curve (only for binary classification settings)
    """
    return 0.0

def evaluate_mse_mae(predictions, ground_truth):
    """
    Compute MSE / MAE for predicted rating values (for rating prediction tasks)
    """
    return {
        "mse": 0.0,
        "mae": 0.0
    }
