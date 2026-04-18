from pathlib import Path
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# -------------------------
# Load dataset
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
input_file = BASE_DIR / "data" / "dili_clean.csv"

df = pd.read_csv(input_file)
print("Loaded rows:", len(df))

# -------------------------
# Convert SMILES to graph
# -------------------------
def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs()
    ]

def smiles_to_data(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) == 0:
        return None

    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor([label], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)

graphs = []
for _, row in df.iterrows():
    g = smiles_to_data(row["smiles"], row["label"])
    if g is not None:
        graphs.append(g)

print("Usable graphs:", len(graphs))

# -------------------------
# Split
# -------------------------
labels = [int(g.y.item()) for g in graphs]
indices = list(range(len(graphs)))

train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

train_labels = [labels[i] for i in train_idx]
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=train_labels, random_state=42)

train_graphs = [graphs[i] for i in train_idx]
val_graphs = [graphs[i] for i in val_idx]
test_graphs = [graphs[i] for i in test_idx]

print("Train:", len(train_graphs))
print("Validation:", len(val_graphs))
print("Test:", len(test_graphs))

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

# -------------------------
# GraphSAGE Model
# -------------------------
class SAGENet(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x.view(-1)

# -------------------------
# Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

in_dim = train_graphs[0].x.shape[1]
model = SAGENet(in_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# -------------------------
# Train
# -------------------------
def train_one_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    y_true, y_prob = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        prob = torch.sigmoid(out).cpu().numpy()
        true = batch.y.view(-1).cpu().numpy()

        y_prob.extend(prob.tolist())
        y_true.extend(true.tolist())

    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "ACC": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

# -------------------------
# Training loop
# -------------------------
best_model = None
best_auc = -1

for epoch in range(1, 21):
    loss = train_one_epoch()
    val_res = evaluate(val_loader)

    print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val AUROC: {val_res['AUROC']:.4f}")

    if val_res["AUROC"] > best_auc:
        best_auc = val_res["AUROC"]
        best_model = copy.deepcopy(model.state_dict())

model.load_state_dict(best_model)

# -------------------------
# Final test results
# -------------------------
test_res = evaluate(test_loader)

print("\nFinal GraphSAGE Test Results")
for k, v in test_res.items():
    print(f"{k}: {v:.4f}")