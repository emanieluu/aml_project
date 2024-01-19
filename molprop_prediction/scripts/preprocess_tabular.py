import pandas as pd
from molprop_prediction.scripts.functions_preprocess_tabular import full_pipeline
from molprop_prediction.scripts.utils import read_train_data, read_test_data

if __name__ == "__main__":  

    train_data = read_train_data()
    test_data = read_test_data()
    X_train_raw, y_train = train_data.drop(columns=["y"]), train_data[["y"]]
    X_test_raw, y_test = test_data.drop(columns=["y"]), test_data[["y"]]
    X_train, X_test = full_pipeline(X_train_raw, X_test_raw, y_train)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train_df = pd.DataFrame(y_train, columns=["y"])
    y_test_df = pd.DataFrame(y_test, columns=["y"])
    preprocessed_train = pd.concat([X_train, y_train_df], axis=1)
    preprocessed_test = pd.concat([X_test, y_test_df], axis=1)
    preprocessed_train.to_csv("./data/preprocessed_data/train_tabular_data.csv")
    preprocessed_test.to_csv("./data/preprocessed_data/test_tabular_data.csv")

