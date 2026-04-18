from pathlib import Path
import pandas as pd
from pubchempy import get_compounds
import time

BASE_DIR = Path(__file__).resolve().parent.parent
input_file = BASE_DIR / "data" / "dili_binary.csv"
output_file = BASE_DIR / "data" / "dili_with_smiles.csv"

df = pd.read_csv(input_file)

def get_smiles(drug_name):
    try:
        compounds = get_compounds(drug_name, "name")
        if compounds and compounds[0].canonical_smiles:
            return compounds[0].canonical_smiles
        return None
    except Exception:
        return None

smiles_list = []

for i, drug in enumerate(df["drug_name"]):
    smiles = get_smiles(drug)
    smiles_list.append(smiles)
    print(f"{i+1}/{len(df)} - {drug} -> {smiles}")
    time.sleep(0.2)

df["smiles"] = smiles_list
df.to_csv(output_file, index=False)

print("\nSaved:", output_file)
print("\nFirst 5 rows:")
print(df.head().to_string())
print("\nMissing SMILES:", df["smiles"].isna().sum())