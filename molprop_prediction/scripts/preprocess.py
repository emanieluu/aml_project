import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx


def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)  # construct molecules from smile notations
    if mol is None:
        return None

    # Convertir la molécule RDKit en graphe NetworkX
    g = nx.Graph()
    for atom in mol.GetAtoms():  # loop over the atoms
        g.add_node(atom.GetIdx(), atom_type=atom.GetSymbol())

    for bond in mol.GetBonds():  # loop over the bonds
        g.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=bond.GetBondTypeAsDouble(),
        )

    return g


class MolDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        mol_graph = smiles_to_graph(row["smiles"])

        x = torch.tensor(
            [[0.0] for _ in range(len(mol_graph.nodes))], dtype=torch.float32
        )

        # Construct edge_index from the adjacency matrix of the graph
        edge_index = (
            torch.tensor(list(mol_graph.edges), dtype=torch.long).t().contiguous()
        )

        # Utilisez row['target_property'] pour obtenir la propriété cible
        target_property = torch.tensor(row["y"], dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, y=target_property)
