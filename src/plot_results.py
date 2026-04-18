from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
input_file = BASE_DIR / "data" / "final_model_comparison.csv"

df = pd.read_csv(input_file)

# Sort for consistency
df = df.sort_values(by="AUROC", ascending=False)

def plot_metric(metric_name, filename, ymin, ymax):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df["Model"], df[metric_name])

    plt.title(f"Model Comparison ({metric_name})")
    plt.xlabel("Models")
    plt.ylabel(metric_name)

    plt.ylim(ymin, ymax)

    for bar, value in zip(bars, df[metric_name]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    output_file = BASE_DIR / "data" / filename
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"{metric_name} chart saved to:", output_file)

# -------------------------
# Create ALL charts
# -------------------------

plot_metric("AUROC", "auroc_chart.png", 0.68, 0.69)
plot_metric("ACC", "accuracy_chart.png", 0.60, 0.66)
plot_metric("F1", "f1_chart.png", 0.72, 0.76)
plot_metric("MCC", "mcc_chart.png", 0.00, 0.25)