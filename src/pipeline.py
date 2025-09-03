import pandas as pd
import yaml
from src.data_preparation.data_cleaning import handle_missing_values, handle_duplicate_values, detect_outliers
from src.data_preparation import dataset_balancing, dataset_splitting
from src.fairness_algorithm import pre_process, in_process, post_process
from src.evaluation import quality_metrics, fairness_metrics

def run_pipeline(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config["data_path"])
    df = handle_duplicate_values(df)
    df = handle_missing_values(df, method=config["data_cleaning"]["fill_missing"])
    df = detect_outliers(df, method=config["data_cleaning"]["outlier_method"])

    if config.get("balancing", {}).get("enable", False):
        method = config["balancing"].get("method", "random")
        if method == "random":
            df = dataset_balancing.random_sampling(df)
        elif method == "cleaning+sampling":
            df = dataset_balancing.cleaning_plus_sampling(df)
        elif method == "cluster":
            df = dataset_balancing.cluster_based_sampling(df)

    # Pre-process (COMPULSORY)
    pre_method = config["pre_process"]["method"]
    if pre_method == "user_activity":
        df = pre_process.group_by_user_activity(df)
    elif pre_method == "item_popularity":
        df = pre_process.group_by_item_popularity(df)
    else:
        raise ValueError("pre_process.method must be 'user_activity' or 'item_popularity'")

    # Split
    train_df, val_df, test_df = dataset_splitting.split_data(df, config["split"])

    # In-process train & predict
    model_name = config["model"]["name"]
    preds_df = in_process.train_and_predict(model_name, config, train_df, val_df, test_df)

    # Post-process
    if config.get("post_process", {}).get("enable", False):
        preds_df = post_process.apply_reranking(preds_df, config)

    # Evaluation (use top-k across all test users)
    topk = config.get("evaluation", {}).get("topk", 20)
    quality = {
        "Precision@K": quality_metrics.evaluate_precision_at_k(preds_df, test_df, k=topk),
        "Recall@K": quality_metrics.evaluate_recall_at_k(preds_df, test_df, k=topk),
        "NDCG@K": quality_metrics.evaluate_ndcg_at_k(preds_df, test_df, k=topk)
    }
    fairness = {
        "Gini": fairness_metrics.evaluate_gini_index(preds_df),
        "KL": fairness_metrics.evaluate_kl_divergence(preds_df, test_df),
        "Recall-Disp": fairness_metrics.evaluate_recall_dispersion(preds_df, test_df)
    }

    return {"quality": quality, "fairness": fairness, "predictions": preds_df}