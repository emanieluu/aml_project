import argparse
import json
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from molprop_prediction.models.GIN import GIN
from molprop_prediction.scripts.preprocess import MolDataset
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


# Paramètres du modèle et de l'entraînement
args = parse_args()
params = load_params(args.config)
input_dim = params["input_dim"]
hidden_dim = params["hidden_dim"]
output_dim = params["output_dim"]
lr = params["lr"]
epochs = params["epochs"]
batch_size = params["batch_size"]

# Initialisation du modèle, de l'optimiseur et de la fonction de perte
train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=42)

# Boucle d'entraînement
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        x, edge_index, batch_data = batch.x, batch.edge_index, batch.batch
        output = model(x, edge_index, batch_data)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}, Loss: {average_loss}")
