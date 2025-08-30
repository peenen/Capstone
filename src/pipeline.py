# src/pipeline.py
from src.data_preparation import data_cleaning, dataset_balancing, dataset_splitting
from src.fairness_algorithm import pre_process, post_process
from src.in_process import train_and_predict
from src.evaluation import quality_metrics, fairness_metrics

def run_pipeline(config):
    """
    Full recommendation pipeline:
    Data cleaning -> Optional balancing -> Optional pre-processing -> In-processing (train+predict)
    -> Optional post-processing -> Evaluation
    """

    # ========= Step 1: Load data =========
    import pandas as pd
    df = pd.read_csv(config["data_path"])
    print("[Step 1] Dataset loaded.")

    # ========= Step 2: Data Cleaning =========
    df = data_cleaning.handle_duplicate_values(df)
    df = data_cleaning.handle_missing_values(df, method=config["data_cleaning"]["fill_missing"])
    df = data_cleaning.detect_outliers(df, method=config["data_cleaning"]["outlier_method"])
    print("[Step 2] Data cleaned.")

    # ========= Step 3: Optional Data Balancing =========
    if config["balancing"].get("enable", False):
        method = config["balancing"].get("method", "random")
        if method == "random":
            df = dataset_balancing.random_sampling(df)
        elif method == "cleaning+sampling":
            df = dataset_balancing.cleaning_plus_sampling(df)
        elif method == "cluster":
            df = dataset_balancing.cluster_based_sampling(df)
        print(f"[Step 3] Balancing applied: {method}")

    # ========= Step 4: Optional Pre-processing =========
    if config["pre_process"].get("enable", False):
        method = config["pre_process"].get("method", "user_activity")
        if method == "user_activity":
            df = pre_process.group_by_user_activity(df)
        elif method == "item_popularity":
            df = pre_process.group_by_item_popularity(df)
        print(f"[Step 4] Pre-processing applied: {method}")

    # ========= Step 5: Data Splitting =========
    train_df, val_df, test_df = dataset_splitting.split_data(df, config["split"])
    print("[Step 5] Data split into train/val/test.")

    # ========= Step 6: In-processing (train + predict) =========
    predictions = train_and_predict(
        train_df,
        test_df,
        model_type=config["model"]["name"],
        in_method=config["model"].get("in_method", None),
        config=config
    )
    print("[Step 6] In-processing complete (predictions generated).")

    # ========= Step 7: Optional Post-processing =========
    if config["post_process"].get("enable", False):
        predictions = post_process.apply_reranking(predictions, config)
        print("[Step 7] Post-processing applied.")

    # ========= Step 8: Evaluation =========
    quality = {
        "F1@20": quality_metrics.evaluate_f1_at_k(predictions, test_df, k=20),
        "NDCG@20": quality_metrics.evaluate_ndcg_at_k(predictions, test_df, k=20)
    }
    fairness = {
        "Gini": fairness_metrics.evaluate_gini_index(predictions),
        "KL": fairness_metrics.evaluate_kl_divergence(predictions, test_df),
        "Recall-Disp": fairness_metrics.evaluate_recall_dispersion(predictions)
    }
    print("[Step 8] Evaluation complete.")

    return {"quality": quality, "fairness": fairness}