from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
input_file = BASE_DIR / "data" / "dili_with_smiles.csv"
output_file = BASE_DIR / "data" / "dili_clean.csv"

df = pd.read_csv(input_file)

print("Original shape:", df.shape)

# Remove rows with missing SMILES
df = df.dropna(subset=["smiles"]).copy()

# Remove empty-string SMILES just in case
df["smiles"] = df["smiles"].astype(str).str.strip()
df = df[df["smiles"] != ""].copy()

# Remove duplicate SMILES
df = df.drop_duplicates(subset=["smiles"]).copy()

# Reset index
df = df.reset_index(drop=True)

# Save cleaned file
df.to_csv(output_file, index=False)

print("\nSaved:", output_file)
print("\nCleaned shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nLabel distribution:")
print(df["label"].value_counts())