import argparse
import json
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from molprop_prediction.models.GIN import GIN
from molprop_prediction.scripts.preprocess_bis import (
    graph_datalist_from_smiles_and_labels,
)
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement du modèle GIN")
    # Supprimez la ligne qui demande le chemin à l'utilisateur
    args = parser.parse_args()
    # Définissez directement le chemin du fichier JSON
    args.config = "molprop_prediction/configs/params.json"
    return args


def load_params(config_path):
    with open(config_path, "r") as config_file:
        params = json.load(config_file)
    return params


# Loading data
merged_data = pd.read_csv("./data/raw_data/train_merged_data.csv")
train_data, test_data = train_test_split(
    merged_data, test_size=0.2, random_state=42
)

train_dataset = graph_datalist_from_smiles_and_labels(
    train_data["smiles"], train_data["y"]
)
test_dataset = graph_datalist_from_smiles_and_labels(
    test_data["smiles"], test_data["y"]
)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Paramètres du modèle et de l'entraînement
args = parse_args()
params = load_params(args.config)
input_dim = train_dataset[0].x.size(-1)
hidden_dim = params["hidden_dim"]
output_dim = params["output_dim"]
lr = params["lr"]
epochs = params["epochs"]
batch_size = params["batch_size"]

# Initialisation du modèle, de l'optimiseur et de la fonction de perte
model = GIN(hidden_dim, input_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
mae = nn.L1Loss()

# Boucle d'entraînement
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_mae = 0  # Nouvelle variable pour stocker la somme des MAE
    for batch in train_dataloader:
        optimizer.zero_grad()
        x, edge_index, batch_data = batch.x, batch.edge_index, batch.batch
        output = model(x, edge_index, batch_data)
        loss = criterion(output, batch.y.view(-1, 1))
        mae_value = mae(output, batch.y.view(-1, 1))  # Calcul de la MAE
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mae += mae_value.item()

    average_loss = total_loss / len(train_dataloader)
    average_mae = total_mae / len(train_dataloader)

    print(f"Epoch {epoch + 1}, Loss: {average_loss}, MAE: {average_mae}")
