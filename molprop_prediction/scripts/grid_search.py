import argparse
import json
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch_geometric.loader import DataLoader
from molprop_prediction.models.GIN import GIN
from molprop_prediction.scripts.preprocess_bis import (
    graph_datalist_from_smiles_and_labels,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement du modèle GIN")
    args = parser.parse_args()
    args.config = "molprop_prediction/configs/params.json"
    return args


def load_params(config_path):
    with open(config_path, "r") as config_file:
        params = json.load(config_file)
    return params


def train_model(model, train_dataloader, optimizer, criterion, mae, device):
    model.train()
    total_loss = 0
    total_mae = 0

    for batch in train_dataloader:
        optimizer.zero_grad()
        x, edge_index, batch_data = batch.x, batch.edge_index, batch.batch
        output = model(x, edge_index, batch_data)
        loss = criterion(output, batch.y.view(-1, 1))
        mae_value = mae(output, batch.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mae += mae_value.item()

    average_loss = total_loss / len(train_dataloader)
    average_mae = total_mae / len(train_dataloader)

    return average_loss, average_mae


def grid_search(train_dataloader, test_dataloader, input_dim, output_dim, device):
    param_grid = {
        "hidden_dim": [64, 128, 256],
        "lr": [0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64],
        "epochs": [20, 30, 40, 50, 80, 100],
    }

    best_model = None
    best_loss = float("inf")

    for params in ParameterGrid(param_grid):
        model = GIN(params["hidden_dim"], input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.MSELoss()
        mae = nn.L1Loss()

        print(f"Testing parameters: {params}")

        for epoch in range(params["epochs"]):
            model.train()
            total_loss = 0
            total_mae = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                x, edge_index, batch_data = batch.x, batch.edge_index, batch.batch
                output = model(x, edge_index, batch_data)
                loss = criterion(output, batch.y.view(-1, 1))
                mae_value = mae(output, batch.y.view(-1, 1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_mae += mae_value.item()

            average_loss = total_loss / len(train_dataloader)
            average_mae = total_mae / len(train_dataloader)

            print(f"Epoch {epoch + 1}, Loss: {average_loss}, MAE: {average_mae}")

        # Evaluate on the test set after training
        model.eval()
        total_loss = 0
        for batch in test_dataloader:
            x, edge_index, batch_data = batch.x, batch.edge_index, batch.batch
            output = model(x, edge_index, batch_data)
            loss = criterion(output, batch.y.view(-1, 1))
            total_loss += loss.item()

        average_loss = total_loss / len(test_dataloader)

        print(f"Test Loss after {params['epochs']} epochs: {average_loss}\n")

        # Update the best model if the current one is better
        if average_loss < best_loss:
            best_loss = average_loss
            best_model = model

    return best_model


def main():
    args = parse_args()
    params = load_params(args.config)

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

    train_dataloader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=params["batch_size"], shuffle=True
    )

    input_dim = train_dataset[0].x.size(-1)
    output_dim = params["output_dim"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model = grid_search(
        train_dataloader, test_dataloader, input_dim, output_dim, device
    )

    print("Best Model:", best_model)


if __name__ == "__main__":
    main()
