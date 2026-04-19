# DILI-Graph-ML
Graph-based machine learning models for DILI prediction

## Overview
This project focuses on predicting Drug-Induced Liver Injury (DILI) using graph-based machine learning models. Each drug molecule is represented as a graph where atoms are nodes and chemical bonds are edges.

The objective is to compare multiple Graph Neural Network (GNN) models and evaluate their performance in toxicity prediction.

---

## Dataset
- Dataset used: DILIrank 2.0 (FDA)
- Final cleaned dataset size: 881 compounds

### Preprocessing Steps
- Cleaned the dataset and selected relevant columns
- Converted labels into binary format:
  - 1 = DILI concern
  - 0 = No DILI concern
- Retrieved SMILES using PubChem
- Removed missing or invalid SMILES

---

## Methodology

### Molecular Representation
- Nodes represent atoms  
- Edges represent chemical bonds  

### Node Features
- Atomic number  
- Degree  
- Formal charge  
- Aromaticity  
- Number of hydrogens  

---

## Models Implemented
The following graph-based models were implemented:

- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE
- GIN (Graph Isomorphism Network)
- MPNN (Message Passing Neural Network)

---

## Evaluation Metrics
The models were evaluated using:

- AUROC (Area Under ROC Curve)
- Accuracy
- F1-score
- MCC (Matthews Correlation Coefficient)

---

## Results

| Model      | AUROC | Accuracy | F1     | MCC    |
|------------|-------|----------|--------|--------|
| GCN        | 0.6859 | 0.6364  | 0.7419 | 0.1831 |
| GAT        | 0.6835 | 0.6477  | 0.7480 | 0.2141 |
| GraphSAGE  | 0.6866 | 0.6080  | 0.7544 | 0.0237 |
| GIN        | 0.6858 | 0.6420  | 0.7249 | 0.2228 |
| MPNN       | 0.6878 | 0.6420  | 0.7342 | 0.2110 |

---

## Visualizations
The following plots were generated:
- AUROC comparison
- Accuracy comparison
- F1-score comparison
- MCC comparison

---

## Technologies Used
- Python  
- PyTorch  
- PyTorch Geometric  
- RDKit  
- Pandas  
- Scikit-learn  

---

## How to Run

### Option 1: Using Conda (Recommended)

Step 1: Create environment  
conda create -n dili_env python=3.10  

Step 2: Activate environment  
conda activate dili_env  

Step 3: Install Python packages  
pip install -r requirements.txt  

Step 4: Install RDKit  
conda install -c conda-forge rdkit  

---

### Option 2: Using pip

Step 1: Create virtual environment  
python -m venv venv  

Step 2: Activate environment (Windows)  
venv\Scripts\activate  

Step 3: Install dependencies  
pip install -r requirements.txt  

---
###Run models:

python src/step6_train_gcn.py

python src/step7_train_gat.py

python src/step8_train_graphsage.py

python src/step9_train_gin.py

python src/step10_train_mpnn.py


---

## Project Structure
dili_project/
│
├── data/
├── src/
├── README.md
└── .gitignore

---

## Conclusion
Graph Neural Networks are effective for predicting drug-induced liver injury. Among the models, MPNN achieved the highest AUROC, while GIN showed balanced performance across metrics.

---

## Author
Md Arshad Islam

