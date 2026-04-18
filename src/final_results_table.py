import pandas as pd
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent
output_file = BASE_DIR / "data" / "final_model_comparison.csv"

# Results of your 5 models
results = [
    {"Model": "GCN", "AUROC": 0.6859, "ACC": 0.6364, "F1": 0.7419, "MCC": 0.1831},
    {"Model": "GAT", "AUROC": 0.6835, "ACC": 0.6477, "F1": 0.7480, "MCC": 0.2141},
    {"Model": "GraphSAGE", "AUROC": 0.6866, "ACC": 0.6080, "F1": 0.7544, "MCC": 0.0237},
    {"Model": "GIN", "AUROC": 0.6858, "ACC": 0.6420, "F1": 0.7249, "MCC": 0.2228},
    {"Model": "MPNN", "AUROC": 0.6878, "ACC": 0.6420, "F1": 0.7342, "MCC": 0.2110},
]

# Create DataFrame
df = pd.DataFrame(results)

# Sort by AUROC (best model on top)
df = df.sort_values(by="AUROC", ascending=False)

# Print table
print("\nFinal Comparison Table (Sorted by AUROC):\n")
print(df.to_string(index=False))

# Save to CSV
df.to_csv(output_file, index=False)

print(f"\nSaved to: {output_file}")