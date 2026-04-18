from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "DILIrank2.xlsx"

# Read raw sheet without assuming headers
df_raw = pd.read_excel(file_path, header=None)

print("Raw shape:", df_raw.shape)
print("\nFirst 15 rows:\n")
print(df_raw.head(15).to_string())