import yaml
from src.pipeline import run_pipeline

if __name__=="__main__":
    with open("config/config.yaml","r",encoding="utf-8") as f:
        config = yaml.safe_load(f)
    results = run_pipeline(config)
    print("\n=== Evaluation Results ===")
    print("[Quality Metrics]")
    for metric, score in results["quality"].items():
        print(f"{metric}: {score}")
    print("\n[Fairness Metrics]")
    for metric, score in results["fairness"].items():
        print(f"{metric}: {score}")