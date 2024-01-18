import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from molprop_prediction.models.GIN_bis import GIN
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


args = parse_args()
params = load_params(args.config)
device = torch.device("cuda:0")


sweep_config = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "hidden_dim": {"values": [256, 64, 128]},
        "lr": {"values": [0.0001, 0.001, 0.01]},
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [200, 150, 100]},
        "random_seed": {"value": 42},
        "num_gin_layers": {"values": [4, 3, 2]},
        "num_lin_layers": {"value": 1},
    },
}


def train(config=None):
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
    train_loader = DataLoader(
        train_dataloader.dataset, batch_size=params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        test_dataloader.dataset, batch_size=params["batch_size"], shuffle=False
    )

    input_dim = train_dataset[0].x.size(-1)

    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        input_dim = train_dataset[0].x.size(-1)

        model = GIN(
            dim_h=config.hidden_dim,
            num_node_features=input_dim,
            num_gin_layers=config.num_gin_layers,
            num_lin_layers=config.num_lin_layers,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.MSELoss()
        mae = nn.L1Loss()

        for epoch in range(config.epochs):
            model.train()
            total_loss = 0
            total_mae = 0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                x, edge_index, batch_data, y = (
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    batch.y,
                )
                output = model(x, edge_index, batch_data)
                loss = criterion(output, batch.y.view(-1, 1))
                mae_value = mae(output, batch.y.view(-1, 1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_mae += mae_value.item()

            average_loss = total_loss / len(train_loader)
            average_mae = total_mae / len(train_loader)

            wandb.log(
                {
                    "epoch": epoch,
                    "train_acc": average_mae,
                    "train_loss": average_loss,
                }
            )

            print(
                f"Epoch {epoch + 1}, Loss: {average_loss}, MAE: {average_mae}"
            )

            # Validation after each epoch
            model.eval()
            total_loss = 0
            total_mae = 0

            for batch in val_loader:
                batch = batch.to(device)
                x, edge_index, batch_data, y = (
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    batch.y,
                )
                output = model(x, edge_index, batch_data)
                loss = criterion(output, batch.y.view(-1, 1))
                mae_value = mae(output, batch.y.view(-1, 1))
                total_loss += loss.item()
                total_mae += mae_value.item()

            average_loss = total_loss / len(val_loader)
            average_mae = total_mae / len(val_loader)

            wandb.log(
                {
                    "test_mae": average_mae,
                    "test_loss": average_loss,
                }
            )

            print(f"Validation Loss after epoch {epoch + 1}: {average_loss}")


def main():
    sweep_id = wandb.sweep(sweep_config, project="hyperparameter-sweep")
    wandb.agent(sweep_id, train)


if __name__ == "__main__":
    main()
