import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from collections import Counter
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import RobustScaler


def smiles_to_mol(dataset,smiles_column, mol_column):
    """transform smiles molecules into mol type 
    Args:
        dataset (Dataframe): Dataset with smiles molecules
        smiles_column (str): Name of smiles column
        mol_column (str): Name we want to give to our new molecule column
    """
    dataset_return = dataset
    dataset_return[mol_column] = dataset_return[smiles_column].apply(lambda x: Chem.MolFromSmiles(x)) 
    return dataset_return.drop(columns=[smiles_column])

def add_fingerprints_features(dataset, molecule_column):
    """ Add 2048 new columns to dataset corresponding to bits of fingerprints representation from mol type, and 2 features calculated from number of bits
    Args:
        dataset (Dataframe): dataset with mol type's molecules
        molecule_column (str): name of the molecule column in mol type
    """
    dataset_return = dataset
    dataset_return['fps_1'] = [np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)) for mol in dataset_return[molecule_column]]
    dataset_return['num_bits_active']= dataset_return['fps_1'].apply(lambda x: np.sum(x))
    dataset_return['mean_bit_density'] = dataset_return['fps_1'].apply(lambda x: np.mean(x))
    dataset_expanded = pd.DataFrame(dataset_return['fps_1'].tolist(), columns=[f'feature_{i}' for i in range(1, 2049)])
    dataset_to_retuned = pd.concat([dataset, dataset_expanded], axis=1)
    dataset_to_retuned = dataset_to_retuned.drop(columns=['fps_1'])
    return dataset_to_retuned

def add_mol_features(dataset, molecule_column):
    """ Add features from molecules to dataset

    Args:
        dataset (Dataframe): dataset with mol type's molecules
        molecule_column (str): name of the molecule column in mol type
    """
    dataset_return = dataset
    dataset_return[molecule_column] = dataset_return[molecule_column].apply(lambda x: Chem.AddHs(x)) # AddHs function adds H atoms to a MOL (as Hs in SMILES are usualy ignored)
    dataset_return['num_of_atoms'] = dataset_return[molecule_column].apply(lambda x: x.GetNumAtoms())
    dataset_return['num_of_heavy_atoms'] = dataset_return[molecule_column].apply(lambda x: x.GetNumHeavyAtoms())
    return dataset_return

def find_top_atoms(dataset, molecule_column, n=8):
    """_summary_

    Args:
        dataset (Dataframe): Dataset
        molecule_column (str): name of the molecule column in mol type
        n (int): number of top atoms

    Returns:
        list : List of strings of the n most current atoms in molecules of the dataset
    """
    all_atoms = Counter()

    for mol in dataset[molecule_column]:
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            all_atoms[symbol] += 1
    all_atoms['H']= all_atoms['Si']=all_atoms['P']= all_atoms['I']=0
    
    top_atoms = [atom for atom, count in all_atoms.most_common(n)]
    return top_atoms

def number_of_atoms(dataset, mol_column, atom_list):
    """Add columns to dataset with number of each atom in atom_list present in molecules

    Args:
        dataset (Datasframe): Dataset
        mol_column (str): name of the molecule column in mol type
        atom_list (list): List of str 
    """
    dataset_return = dataset
    for i in atom_list:
        dataset_return['num_of_{}_atoms'.format(i)] = dataset_return[mol_column].apply(lambda x: len(x.GetSubstructMatches(Chem.MolFromSmiles(i))))
    return dataset_return



def add_descriptors(dataset, mol_column):
    """ Add columns of descriptors from the RDKit package

    Args:
        dataset (Datasframe): Dataset
        mol_column (str): name of the molecule column in mol type
    """
    descrs = [Descriptors.CalcMolDescriptors(mol) for mol in dataset[mol_column]]
    descrs= pd.DataFrame(descrs)
    dataset_return = pd.concat([dataset, descrs], axis=1)
    return dataset_return

def remoove_na_and_cols(dataset,cols_to_remoove):
    """ Remoove columns with NaN and columns given 

    Args:
        dataset (dataframe): our dataset
        cols_to_remoove (list of str): List of columns names we want to remoove 
    """
    dataset_return = dataset
    na_col = dataset_return.columns[dataset_return.isna().any()].tolist()
    dataset_return = dataset_return.drop(columns= cols_to_remoove + na_col)
    return dataset_return

def add_features(dataset, smiles_column, mol_column, n = 8):
    """ Create our final dataset 

    Args:
        dataset (Dataframe): Dataset with smiles molecules
        smiles_column (str): Name of smiles column
        mol_column (str): Name we want to give to our new molecule column
        n (int): number of top atoms to add 
    """
    dataset_return = smiles_to_mol(dataset, smiles_column, mol_column)
    dataset_return = add_fingerprints_features(dataset_return, mol_column)
    dataset_return = add_mol_features(dataset_return, mol_column)
    # atom_list = find_top_atoms(dataset_return, mol_column, n)
    # dataset_return = number_of_atoms(dataset_return, mol_column, atom_list)
    dataset_return = add_descriptors(dataset_return, mol_column)
    dataset_return = remoove_na_and_cols(dataset_return, ['id', mol_column])
    return dataset_return 

def full_pipeline(X_train, X_test, y_train):
    X_train = add_features(X_train, "smiles","mol")
    X_test = add_features(X_test, "smiles","mol")
    rs = RobustScaler()
    kb = SelectKBest(k=400)
    X_train = rs.fit_transform(X_train)
    X_train = kb.fit_transform(X_train, y_train)
    X_test = rs.transform(X_test)
    X_test = kb.transform(X_test)
    return X_train, X_test

