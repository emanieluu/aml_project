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

################################################################################################

import json
from rdkit import Chem
from rdkit.Chem import GraphDescriptors
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit import Chem
from pathlib import Path
import torch
import numpy as np, pandas as pd
import torch
from torch_geometric.data import Data


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
        "atomic_mass_scaled": lambda atom: [float((atom.GetMass() - 10.812) / 116.092)],
        "vdw_radius_scaled": lambda atom: [
            float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)
        ],
        "covalent_radius_scaled": lambda atom: [
            float(
                (Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)
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
        "bond_types": onehot_encodings(lambda bond: bond.GetBondType(), bond_types),
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

################################################################################################

from collections import defaultdict
import numpy as np
from rdkit import Chem
import torch


def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], "aromatic")
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):
            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)


def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
    np.random.seed(1234)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def create_datasets(task, dataset, radius, device):
    dir_dataset = "../dataset/" + task + "/" + dataset + "/"

    """Initialize x_dict, in which each key is a symbol type
    (e.g., atom and chemical bond) and each value is its index.
    """
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    def create_dataset(filename):
        print(filename)

        """Load a dataset."""
        with open(dir_dataset + filename, "r") as f:
            smiles_property = f.readline().strip().split()
            data_original = f.read().strip().split("\n")

        """Exclude the data contains '.' in its smiles."""
        data_original = [data for data in data_original if "." not in data.split()[0]]

        dataset = []

        for data in data_original:
            smiles, property = data.strip().split()

            """Create each data with the above defined functions."""
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(
                radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict
            )
            adjacency = Chem.GetAdjacencyMatrix(mol)

            """Transform the above each data of numpy
            to pytorch tensor on a device (i.e., CPU or GPU).
            """
            fingerprints = torch.LongTensor(fingerprints).to(device)
            adjacency = torch.FloatTensor(adjacency).to(device)
            if task == "classification":
                property = torch.LongTensor([int(property)]).to(device)
            if task == "regression":
                property = torch.FloatTensor([[float(property)]]).to(device)

            dataset.append((fingerprints, adjacency, molecular_size, property))

        return dataset

    dataset_train = create_dataset("data_train.txt")
    dataset_train, dataset_dev = split_dataset(dataset_train, 0.9)
    dataset_test = create_dataset("data_test.txt")

    N_fingerprints = len(fingerprint_dict)

    return dataset_train, dataset_dev, dataset_test, N_fingerprints
