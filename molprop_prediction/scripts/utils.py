import argparse
import json
import pandas as pd 
import torch
from molprop_prediction.scripts.functions_preprocess_graph import (
    graph_datalist_from_smiles_and_labels,
)
from torch_geometric.loader import DataLoader


def parse_args(arg_path):
    parser = argparse.ArgumentParser(description="Entraînement du modèle GIN")
    args = parser.parse_args()
    args.config = arg_path
    return args


def load_params(config_path):
    with open(config_path, "r") as config_file:
        params = json.load(config_file)
    return params


def prompt_user_for_args():
    model = input("Which model would you like to train? (RF, GIN, GAT): ")
    params_file = input("Enter the name of the JSON parameter file to load (default: params.json): ")
    params_file = params_file.strip() or "params.json"
    saving_name = input("Enter the name of the model (saved in models/trained_models): ")
    print(f"Training the {model} model with parameters from the file {params_file} and saving it as {saving_name}.")
    config_path = f"./molprop_prediction/configs/{model}/{params_file}"
    save_path = f"./molprop_prediction/models/{model}/trained_models/{saving_name}"
    return model, config_path, save_path


def prompt_user_for_predictions():
    model = input("Which model would you like to use to predict? (RF, GIN, GAT): ")
    checkpoint_name = input("Enter the name of the model (saved in models/trained_models): ")
    params_file = input("Enter the name of the JSON parameter file to load (default: params.json): ")
    params_file = params_file.strip() or "params.json"
    config_path = f"./molprop_prediction/configs/{model}/{params_file}"
    checkpoint_path = "./molprop_prediction/models/" + model + "/trained_models/" + checkpoint_name
    print(f"Using {model} model with parameters from the file {params_file} and checkpoint {checkpoint_name} to predict")
    save_path = "./data/predictions/" + model + "_predictions/" + "predictions.csv"
    return model, checkpoint_path, config_path, save_path


def read_train_data():
    train_data = pd.read_csv("./data/raw_data/fixed_train_data.csv")
    train_data.drop(columns="Unnamed: 0")
    return train_data


def read_test_data():
    test_data = pd.read_csv("./data/raw_data/fixed_test_data.csv")
    test_data.drop(columns="Unnamed: 0")
    return test_data


def read_tabular_train():
    data = pd.read_csv("./data/preprocessed_data/tabular_data_train.csv")
    return data 


def read_tabular_test():
    data = pd.read_csv("./data/preprocessed_data/tabular_data_test.csv")
    return data


def preprocess_graph_data(data):
    data = graph_datalist_from_smiles_and_labels(
        data["smiles"], data["y"]
    )
    graph_dataloader = DataLoader(data, batch_size=32, shuffle=True)
    return graph_dataloader 


def load_graph_preprocessed_test_dataset():
    test_data = pd.read_csv("./data/raw_data/_test_fixed.csv")
    test_dataset = graph_datalist_from_smiles_and_labels(test_data["smiles"], test_data["y"])
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    kept_test_id = test_data['id']
    return test_dataloader, kept_test_id


def load_model(model, optimizer, checkpoint_path):
    loaded_checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(loaded_checkpoint['model_state_dict'])
    optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    epoch = loaded_checkpoint['epoch']
    loss = loaded_checkpoint['loss']
    mae = loaded_checkpoint['mae']
    print(f"Model loaded from {checkpoint_path}, trained for {epoch} epochs. Last loss: {loss}, Last MAE: {mae}")
    return model, optimizer


def translate_params(param_indices):
    translated_params_RF = {}
    translated_params_pipeline = {}

    param_mapping = {
        'n_estimators': [50, 100, 200, 300, 400],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 3, 7, 12],
        'min_samples_leaf': [1, 3, 6, 10],
        'bootstrap': [True, False],
        'warm_startbool': [True, False],
        'max_features': ['sqrt', 'log2'],
        'preprocessor': ['StandardScaler', 'RobustScaler', 'Normalizer', 'MaxAbsScaler'],
        'feature_extractor': ['pca', 'RFE', 'SelectKBest'],
        'n_components': [10, 50, 60, 75, 100, 150, 200],
        'whiten': [True, False],
        'n_features_to_select': [200, 300, 400],
        'step': 10,
        'k': [10, 75, 150, 200, 300, 400]
    }

    for param, index in param_indices.items():
        if param == 'feature_extractor'or param=='k'or param=='preprocessor' or param=='n_components' or param=='whiten' or param=='n_features_to_select' or param=='step':
            translated_params_pipeline[param] = param_mapping[param][index]
        else:
            translated_params_RF[param] = param_mapping[param][index]
            translated_params_pipeline[param] = param_mapping[param][index]

    return translated_params_RF, translated_params_pipeline