import argparse
import json
import pandas as pd 
import torch
from sklearn.model_selection import train_test_split
from molprop_prediction.scripts.preprocess import (
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
    params_file = input("Enter the name of the JSON parameter file to load: ")
    model_name = input("Enter the name of the model (saved in models/trained_models): ")
    return model, params_file, model_name


def prompt_user_for_predictions():
    model = input("Which model would you like to use to predict? (RF, GIN, GAT): ")
    checkpoint_name = input("Enter the name of the model (saved in models/trained_models): ")
    params_file = input("Enter the name of the JSON parameter file to load: ")
    return model, checkpoint_name, params_file


def read_train_data():
    train_data = pd.read_csv("./data/raw_data/fixed_train_data.csv", index_col=0)
    return train_data

def read_test_data():
    test_data = pd.read_csv("./data/raw_data/fixed_test_data.csv", index_col=0)
    return test_data

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
