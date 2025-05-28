import yaml
from src.pipeline import run_pipeline

if __name__ == "__main__":
    # 读取配置文件
    # Load configuration file
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 执行推荐系统主流程
    # Run the full recommendation pipeline
    results = run_pipeline(config)

    # 输出评估指标
    # Print evaluation results
    print("\n=== Evaluation Results ===")
    print("[Quality Metrics]")
    for metric, score in results["quality"].items():
        print(f"{metric}: {score}")

    print("\n[Fairness Metrics]")
    for metric, score in results["fairness"].items():
        print(f"{metric}: {score}")
