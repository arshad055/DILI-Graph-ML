from pathlib import Path
import pandas as pd
from rdkit import Chem

BASE_DIR = Path(__file__).resolve().parent.parent
input_file = BASE_DIR / "data" / "dili_clean.csv"

df = pd.read_csv(input_file)

print("Total molecules:", len(df))

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    nodes = []
    for atom in mol.GetAtoms():
        nodes.append(atom.GetAtomicNum())

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))

    return nodes, edges

graphs = []

for i, smiles in enumerate(df["smiles"]):
    result = smiles_to_graph(smiles)

    if result is None:
        graphs.append(None)
    else:
        nodes, edges = result
        graphs.append({"nodes": nodes, "edges": edges})

    if i < 5:
        print(f"\nExample {i+1}")
        print("SMILES:", smiles)
        print("Nodes:", nodes)
        print("Edges:", edges)

print("\nFinished converting SMILES to graphs")

valid_graphs = sum([1 for g in graphs if g is not None])
print("Valid graphs:", valid_graphs)