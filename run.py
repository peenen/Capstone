from src.pipeline import run_pipeline
import yaml

if __name__ == "__main__":
    results = run_pipeline("config/config.yaml")
    print("\n=== Quality Metrics ===")
    for k,v in results["quality"].items():
        print(f"{k}: {v:.6f}")
    print("\n=== Fairness Metrics ===")
    for k,v in results["fairness"].items():
        print(f"{k}: {v:.6f}")

    preds = results.get("predictions")
    if preds is not None and not preds.empty:
        print("\n=== Sample top recommendations (first 100 rows) ===")
        print(preds.sort_values(['user_id','prediction'], ascending=[True, False]).head(100).to_string(index=False))