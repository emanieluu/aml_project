import numpy as np
import pandas as pd
import torch
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
from rdkit.Chem import GraphDescriptors
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from pathlib import Path


def smiles_to_graph(smiles):
    """
    Converts a SMILES string to an RDKit molecule object.
    SMILES strings are a textual representation of molecular structures.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Initialize list to store graph data
    node_features = []
    edge_indices = []

    for atom in mol.GetAtoms():
        node_features.append([atom.GetAtomicNum()])

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [
            [i, j],
            [j, i],
        ]  # Ajouter les deux directions pour le graphe non orienté

    # Convert node features and edge indices to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


class MolDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        graph = smiles_to_graph(row["smiles"])
        if graph is None:
            return None
        graph.y = torch.tensor([row["y"]], dtype=torch.float32)
        return graph


def onehot_encode(x, features: list):
    """
    Maps input elements x not in features to the last element
    """
    if x not in features:
        x = features[-1]
    binary_encoding = [
        int(bool_val) for bool_val in list(map(lambda s: x == s, features))
    ]
    return binary_encoding


class onehot_encodings:
    """encoding class for one hot features"""

    def __init__(self, atom_info_func, features):
        self.atom_info_func = atom_info_func
        self.features = features

    # @property
    def onehot_encodings(self, atom):
        return onehot_encode(self.atom_info_func(atom), self.features)

    def __len__(self):
        return len(self.features)


def json_to_list(json_file):
    with open(json_file, "r") as f:
        input_data = list(json.load(f).values())
    print(f"Loaded json from: {json_file}")
    return input_data


def list_to_json(json_file, input_list):
    with open(json_file, "w") as f:
        json.dump(input_list, f)
    print(f"Output json saved to: {json_file}")


class FeaturesArgs:
    # encodings information
    available_atoms = [
        "C",
        "N",
        "O",
        "S",
        "F",
        "Si",
        "P",
        "Cl",
        "Br",
        "Mg",
        "Na",
        "Ca",
        "Fe",
        "As",
        "Al",
        "I",
        "B",
        "V",
        "K",
        "Tl",
        "Yb",
        "Sb",
        "Sn",
        "Ag",
        "Pd",
        "Co",
        "Se",
        "Ti",
        "Zn",
        "Li",
        "Ge",
        "Cu",
        "Au",
        "Ni",
        "Cd",
        "In",
        "Mn",
        "Zr",
        "Cr",
        "Pt",
        "Hg",
        "Pb",
        "Unknown",
    ]
    chirality = [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ]
    num_hydrogens = [0, 1, 2, 3, 4, "MoreThanFour"]
    n_heavy_atoms = num_hydrogens
    formal_charges = [-3, -2, -1, 0, 1, 2, 3, "Extreme"]
    hybridisation_type = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
    # Atoms
    # atom encodings
    atom_encoding_lambdas = {
        "available_atoms": onehot_encodings(
            lambda atom: str(atom.GetSymbol()), available_atoms
        ),
        "chirality_type_enc": onehot_encodings(
            lambda atom: str(atom.GetChiralTag()), chirality
        ),
        "hydrogens_implicit": onehot_encodings(
            lambda atom: int(atom.GetTotalNumHs()), num_hydrogens
        ),
        "n_heavy_atoms": onehot_encodings(
            lambda atom: int(atom.GetDegree()), n_heavy_atoms
        ),
        "formal_charge": onehot_encodings(
            lambda atom: int(atom.GetFormalCharge()), formal_charges
        ),
        "hybridisation_type": onehot_encodings(
            lambda atom: str(atom.GetHybridization()), hybridisation_type
        ),
    }
    # atom info
    atom_info_lambdas = {
        "is_in_a_ring_enc": lambda atom: [int(atom.IsInRing())],
        "is_aromatic_enc": lambda atom: [int(atom.GetIsAromatic())],
        "atomic_mass_scaled": lambda atom: [
            float((atom.GetMass() - 10.812) / 116.092)
        ],
        "vdw_radius_scaled": lambda atom: [
            float(
                (Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)
                / 0.6
            )
        ],
        "covalent_radius_scaled": lambda atom: [
            float(
                (
                    Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())
                    - 0.64
                )
                / 0.76
            )
        ],
    }
    # compute node feature length
    n_node_features = sum(map(len, atom_encoding_lambdas.values()))
    n_node_features += len(atom_info_lambdas)

    # Bonds encoding info
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    stereo_types = ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]
    # bond encodings
    bond_encoding_lambdas = {
        "bond_types": onehot_encodings(
            lambda bond: bond.GetBondType(), bond_types
        ),
        "stereo_types": onehot_encodings(
            lambda bond: str(bond.GetStereo()), stereo_types
        ),
    }
    # bond quantity
    bond_info_lambas = {
        "bond_is_conj_enc": lambda bond: [int(bond.GetIsConjugated())],
        "bond_is_in_ring_enc": lambda bond: [int(bond.IsInRing())],
    }
    n_edge_features = sum(map(len, bond_encoding_lambdas.values()))
    n_edge_features += len(bond_info_lambas)
    #
    n_features = n_edge_features + n_node_features
    # Molecule
    # lambda mol: GraphDescriptors.BalabanJ(mol)


class ModelArgs:
    available_models = ["GIN"]
    model = "GIN"


class TrainArgs:
    batch_size = 2**5
    lr = 5e-3
    weight_decay = 5e-4
    epochs = 20
    name = "default-GIN"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_save_pt = Path(f"./model/{name}.pth")


class InferArgs:
    batch_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "default-GIN"
    output_path = Path("./output")
    model_save_pt = Path(f"./model/{name}.pth")


class DataArgs:
    data_source = Path("data/kinase_JAK.csv")
    num_workers = 4
    device = TrainArgs.device


def get_atom_features(
    atom,
    available_atoms: list = FeaturesArgs.available_atoms,
    atom_encode_lambdas: dict = FeaturesArgs.atom_encoding_lambdas,
    atom_info_lambdas: dict = FeaturesArgs.atom_info_lambdas,
    debug=False,
):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    if "hydrogens_implicit" in atom_encode_lambdas:
        available_atoms = ["H"] + available_atoms
    atom_feature_vector = []
    # compute atom features
    for name, atom_encoding in atom_encode_lambdas.items():
        encoding = atom_encoding.onehot_encodings(atom)
        atom_feature_vector += encoding
        if debug:
            print(f"atom encoding length ({name}): {len(encoding)}")
        # boolean features
    for name, info_func in atom_info_lambdas.items():
        atom_feature_vector += info_func(atom)
        if debug:
            print(f"atom info ({name}): {info_func(atom)}")
    if debug:
        print(f"full atom feature:{len(atom_feature_vector)}")
    return torch.Tensor(atom_feature_vector)


def get_bond_features(
    bond,
    bond_encoding_lambdas: dict = FeaturesArgs.bond_encoding_lambdas,
    bond_info_lambdas: dict = FeaturesArgs.bond_info_lambas,
    debug=False,
):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    bond_feature_vector = []
    # compute bond features
    for name, bond_encoding in bond_encoding_lambdas.items():
        encoding = bond_encoding.onehot_encodings(bond)
        bond_feature_vector += encoding
        if debug:
            print(f"bond encoding length ({name}): {len(bond_feature_vector)}")
        # boolean features
    for name, info_func in bond_info_lambdas.items():
        bond_feature_vector += info_func(bond)
        if debug:
            print(f"bond info ({name}): {info_func(bond)}")
        return torch.Tensor(bond_feature_vector)


def smile_to_data(smiles, y_val):
    """smile to pyg Data components"""
    # convert SMILES to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)
    # get feature dimensions
    n_nodes = mol.GetNumAtoms()
    n_edges = 2 * mol.GetNumBonds()

    # construct node feature matrix X of shape (n_nodes, n_node_features)
    for n, atom in enumerate(mol.GetAtoms()):
        atom_features = get_atom_features(atom)
        if n == 0:
            X = torch.zeros((n_nodes, len(atom_features)), dtype=torch.float)
        X[atom.GetIdx(), :] = atom_features

    # construct edge index array E of shape (2, n_edges)
    E_ij = torch.stack(
        list(
            map(
                lambda arr: torch.Tensor(arr).to(torch.long),
                np.nonzero(GetAdjacencyMatrix(mol)),
            )
        )
    )
    # construct edge feature array EF of shape (n_edges, n_edge_features)
    EF = torch.stack(
        [
            get_bond_features(mol.GetBondBetweenAtoms(i.item(), j.item()))
            for i, j in zip(E_ij[0], E_ij[1])
        ]
    )
    # construct label tensor
    y_tensor = torch.tensor(np.array([y_val]), dtype=torch.float)
    return X, E_ij, EF, y_tensor


def graph_datalist_from_smiles_and_labels(x_smiles, y):
    """
    Inputs:
      x_smiles [list]: SMILES strings
      y [list]: numerial labels for the SMILES strings
    Outputs:
      data_list [list]: torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning

    """
    data_list = []
    for smiles, y_val in zip(x_smiles, y):
        X, E, EF, y_tensor = smile_to_data(smiles, y_val)
        # construct Pytorch Geometric data object list
        data_list.append(Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor))
    return data_list
