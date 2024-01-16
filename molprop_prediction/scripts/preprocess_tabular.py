from rdkit import Chem
import numpy as np
import pandas as pd
from collections import Counter
from rdkit.Chem import AllChem
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class SmilesToMolTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, smiles_column):
        self.smiles_column = smiles_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['mol'] = X_copy[self.smiles_column].apply(lambda x: Chem.MolFromSmiles(x))
        return X_copy.drop(columns=[self.smiles_column])

class AddFingerprintsFeaturesTransformer(BaseEstimator, TransformerMixin):
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
        dataset_to_returned = pd.concat([X_copy, dataset_expanded], axis=1)
        dataset_to_returned = dataset_to_returned.drop(columns=['fps_1'])
        return dataset_to_returned

class AddMolFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, molecule_column):
        self.molecule_column = molecule_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        def add_hs(mol):
            try:
                return Chem.AddHs(mol)
            except:
                return None

        X_copy[self.molecule_column] = X_copy[self.molecule_column].apply(lambda x: add_hs(x))
        
        # Filter out rows where Chem.AddHs failed
        X_copy = X_copy.dropna(subset=[self.molecule_column])

        X_copy['num_of_atoms'] = X_copy[self.molecule_column].apply(lambda x: x.GetNumAtoms())
        X_copy['num_of_heavy_atoms'] = X_copy[self.molecule_column].apply(lambda x: x.GetNumHeavyAtoms())
        return X_copy


class FindTopAtomsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, molecule_column, n):
        self.molecule_column = molecule_column
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        all_atoms = Counter()
        for mol in X[self.molecule_column]:
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                all_atoms[symbol] += 1
        all_atoms['H'] = all_atoms['Si'] = 0
        top_atoms = [atom for atom, count in all_atoms.most_common(self.n)]
        return top_atoms

def create_feature_pipeline():
    feature_pipeline = Pipeline([
    ('smiles_to_mol', SmilesToMolTransformer(smiles_column='smiles')),
    ('add_fingerprints', AddFingerprintsFeaturesTransformer(molecule_column='mol')),
    ('add_mol_features', AddMolFeaturesTransformer(molecule_column='mol')),
    ('find_top_atoms', FindTopAtomsTransformer(molecule_column='mol', n=5))])
    return feature_pipeline

