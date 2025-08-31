import pandas as pd
from src.data_preparation import data_cleaning, dataset_balancing, dataset_splitting
from src.fairness_algorithm import pre_process, in_process, post_process
from src.models.baseline_models import MFModel, LightGCNModel
from src.evaluation import quality_metrics, fairness_metrics

def run_pipeline(config):
    df = pd.read_csv(config["data_path"])
    df = data_cleaning.handle_duplicate_values(df)
    df = data_cleaning.handle_missing_values(df, method=config["data_cleaning"]["fill_missing"])
    df = data_cleaning.detect_outliers(df, method=config["data_cleaning"]["outlier_method"])

    # Optional balancing
    if config["balancing"]["enable"]:
        method = config["balancing"]["method"]
        if method=="random":
            df = dataset_balancing.random_sampling(df)
        elif method=="cleaning+sampling":
            df = dataset_balancing.cleaning_plus_sampling(df)
        elif method=="cluster":
            df = dataset_balancing.cluster_based_sampling(df)

    # Pre-process (compulsory)
    pre_method = config["pre_process"]["method"]
    if pre_method=="user_activity":
        df = pre_process.group_by_user_activity(df)
    elif pre_method=="item_popularity":
        df = pre_process.group_by_item_popularity(df)

    train_df, val_df, test_df = dataset_splitting.split_data(df, config["split"])

    # Model initialization
    model_name = config["model"]["name"]
    model_cfg = config["model"]
    if model_name=="MF":
        model = MFModel(model_cfg)
    elif model_name=="LightGCN":
        model = LightGCNModel(model_cfg)

    # In-process
    if config["in_process"]["enable"]:
        model.in_method = config["in_process"]["method"]
        predictions = in_process.apply_in_process(model, train_df, val_df)
    else:
        predictions = model.predict(test_df)

    # Post-process
    if config["post_process"]["enable"]:
        predictions = post_process.apply_reranking(predictions, config)

    # Evaluation
    quality = {
        "F1@20": quality_metrics.evaluate_f1_at_k(predictions, test_df),
        "NDCG@20": quality_metrics.evaluate_ndcg_at_k(predictions, test_df)
    }
    fairness = {
        "Gini": fairness_metrics.evaluate_gini_index(predictions),
        "KL": fairness_metrics.evaluate_kl_divergence(predictions, test_df),
        "Recall-Disp": fairness_metrics.evaluate_recall_dispersion(predictions)
    }
    return {"quality": quality, "fairness": fairness}