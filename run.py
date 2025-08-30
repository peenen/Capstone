import yaml
from src.pipeline import run_pipeline

if __name__ == "__main__":
    # Load configuration file
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Run the full recommendation pipeline
    results = run_pipeline(config)

    # Print evaluation results
    print("\n=== Evaluation Results ===")

    if "quality" in results:
        print("[Quality Metrics]")
        for metric, score in results["quality"].items():
            print(f"{metric}: {score:.4f}")

    if "fairness" in results:
        print("\n[Fairness Metrics]")
        for metric, score in results["fairness"].items():
            print(f"{metric}: {score:.4f}")