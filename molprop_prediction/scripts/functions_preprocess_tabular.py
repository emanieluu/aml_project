import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

class SmilesToMol(BaseEstimator, TransformerMixin):
    def __init__(self, smiles_column, mol_column):
        self.smiles_column = smiles_column
        self.mol_column = mol_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.mol_column] = X_copy[self.smiles_column].apply(lambda x: Chem.MolFromSmiles(x))
        return X_copy.drop(columns=[self.smiles_column])

class FingerprintFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, molecule_column):
        self.molecule_column = molecule_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['fps_1'] = [np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)) for mol in X_copy[self.molecule_column]]
        X_copy['num_bits_active'] = X_copy['fps_1'].apply(lambda x: np.sum(x))
        X_copy['mean_bit_density'] = X_copy['fps_1'].apply(lambda x: np.mean(x))
        dataset_expanded = pd.DataFrame(X_copy['fps_1'].tolist(), columns=[f'feature_{i}' for i in range(1, 2049)])
        return pd.concat([X_copy, dataset_expanded], axis=1).drop(columns=['fps_1'])

class MolFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, molecule_column):
        self.molecule_column = molecule_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def add_hydrogens(mol):
            try:
                # Check if mol is a valid RDKit molecule
                if mol is not None and Chem.MolToSmiles(mol) != '':
                    return Chem.AddHs(mol)
                else:
                    return mol
            except:
                return mol

        X_copy = X.copy()
        X_copy[self.molecule_column] = X_copy[self.molecule_column].apply(add_hydrogens)
        X_copy['num_of_atoms'] = X_copy[self.molecule_column].apply(lambda x: x.GetNumAtoms() if x is not None else np.nan)
        X_copy['num_of_heavy_atoms'] = X_copy[self.molecule_column].apply(lambda x: x.GetNumHeavyAtoms() if x is not None else np.nan)
        return X_copy

class DescriptorsFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, molecule_column):
        self.molecule_column = molecule_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        descrs = [Descriptors.CalcMolDescriptors(mol) if mol is not None else [np.nan] * Descriptors._descList.__len__() for mol in X[self.molecule_column]]
        descrs_df = pd.DataFrame(descrs, columns=[desc[0] for desc in Descriptors._descList])
        return pd.concat([X, descrs_df], axis=1)

class RemoveNaAndCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_remove):
        self.cols_to_remove = cols_to_remove

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        na_cols = X.columns[X.isna().any()].tolist()
        return X.drop(columns=self.cols_to_remove + na_cols)

class PreprocessTabular(BaseEstimator, TransformerMixin):
    def __init__(self, smiles_column, mol_column, n=8):
        self.smiles_column = smiles_column
        self.mol_column = mol_column
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dataset_return = SmilesToMol(self.smiles_column, self.mol_column).fit_transform(X)
        dataset_return = FingerprintFeatures(self.mol_column).fit_transform(dataset_return)
        dataset_return = MolFeatures(self.mol_column).fit_transform(dataset_return)
        dataset_return = DescriptorsFeatures(self.mol_column).fit_transform(dataset_return)
        return RemoveNaAndCols(['id', self.mol_column]).fit_transform(dataset_return)
