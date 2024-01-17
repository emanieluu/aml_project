from molprop_prediction.scripts.functions_preprocess_tabular import PreprocessTabular
from molprop_prediction.scripts.utils import read_train_data, read_test_data
from sklearn.pipeline import Pipeline
import pandas as pd

if __name__ == "__main__":   
    train_data = read_train_data()
    test_data = read_test_data()
    X_train, y_train = train_data.drop('y', axis=1), train_data['y']
    X_test, y_test = test_data.drop('y', axis=1), test_data['y']
    pipeline = Pipeline([
        ('PreprocessTabular', PreprocessTabular("smiles", "mol", 8)),
        # Add more pipeline steps if needed
    ])
    X_train = pipeline.fit_transform(train_data)
    X_test = pipeline.transform(test_data)
    preprocessed_train = pd.concat([X_train, y_train], axis=1)
    preprocessed_test = pd.concat([X_test, y_test], axis=1)
    preprocessed_train.to_csv("aml_project/data/preprocessed_data/train_tabular_data.csv")
    preprocessed_test.to_csv("aml_project/data/preprocessed_data/test_tabular_data.csv")
