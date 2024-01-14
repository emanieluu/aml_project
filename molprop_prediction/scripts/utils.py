import argparse
import json
import pandas as pd 
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
    save_path = input("Enter the name of the model (saved in models/trained_models): ")
    return model, params_file, save_path

def load_graph_preprocessed_dataset():
    merged_data = pd.read_csv("./data/raw_data/train_merged_data.csv")
    train_data, test_data = train_test_split(
        merged_data, test_size=0.1, random_state=42
    )

    train_dataset = graph_datalist_from_smiles_and_labels(
        train_data["smiles"], train_data["y"]
    )
    test_dataset = graph_datalist_from_smiles_and_labels(
        test_data["smiles"], test_data["y"]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader


