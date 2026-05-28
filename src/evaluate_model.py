import json
from pathlib import Path


def load_metrics(metrics_path="results/baseline_metrics.json"):
    metrics_file = Path(metrics_path)

    if not metrics_file.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_file}. "
            "Run python src/train_model.py first."
        )

    with open(metrics_file, "r", encoding="utf-8") as file:
        return json.load(file)


def print_metrics(metrics):
    print("Baseline Model Metrics")
    print("----------------------")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


def main():
    metrics = load_metrics()
    print_metrics(metrics)


if __name__ == "__main__":
    main()