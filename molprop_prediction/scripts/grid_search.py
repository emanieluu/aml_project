import argparse
import json
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
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

def grid_search(train_dataloader, test_dataloader, input_dim, output_dim, device):
    param_grid = {       
        "hidden_dim": [256, 64, 128],
        "lr": [0.0001,0.001, 0.01],
        "batch_size": [16, 32, 64],
        "epochs": [200, 150, 100],  #[40, 30],
        "random_seed": [37],
        "k_folds": [3],
        "num_gin_layers": [4, 3, 2],  
        "num_lin_layers": [1],
    }

    best_model = None
    best_loss = float("inf")
    best_params = {}

    results_file = "molprop_prediction/results/grid_search_res_2.txt"
    with open(results_file, 'w') as f:
        f.write("hidden_dim,lr,batch_size,epochs,num_gin_layers,num_lin_layers,val_avg_loss,val_avg_mae,test_avg_loss,test_avg_loss\n")

    for params in ParameterGrid(param_grid):
        print(params)
        kf = KFold(n_splits=params["k_folds"], shuffle=True, random_state=params["random_seed"])
        val_losses = []
        mae_values = []

        for fold, (train_index, val_index) in enumerate(kf.split(train_dataloader.dataset), 1):
            train_set = torch.utils.data.Subset(train_dataloader.dataset, train_index)
            val_set = torch.utils.data.Subset(train_dataloader.dataset, val_index)

            train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True)
            val_loader = DataLoader(val_set, batch_size=params["batch_size"], shuffle=False)

            model = GIN(
                dim_h=params["hidden_dim"],
                num_node_features=input_dim,
                num_gin_layers=params["num_gin_layers"],
                num_lin_layers=params["num_lin_layers"],
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=params["lr"])
            criterion = nn.MSELoss()
            mae = nn.L1Loss()

            print(f"Training on fold {fold}:")
            loss_fold = []
            mae_fold = []

            for epoch in range(params["epochs"]):
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

                print(f"Epoch {epoch + 1}, Loss: {average_loss}, MAE: {average_mae}")
                loss_fold.append(average_loss)
                mae_fold.append(average_mae)

            # Evaluate on the validation set
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

            print(f"Validation Loss after {params['epochs']} epochs: {average_loss}")
            val_losses.append(average_loss)
            mae_values.extend(mae_fold)

        # Average validation loss over folds
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_mae = sum(mae_values) / len(mae_values)
        print(f"Avg. Validation Loss: {avg_val_loss}")
        print(f"Avg. MAE across folds: {avg_mae}")

        # Update the best model if the current one is better
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = model
            best_params = params

        # Evaluate on the test set after training
        model.eval()
        total_loss = 0

        for batch in test_dataloader:
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

        test_avg_loss = total_loss / len(test_dataloader)
        test_avg_mae = total_mae / len(test_dataloader)

        print(f"Test Loss after {params['epochs']} epochs: {average_loss}\n")

        with open(results_file, 'a') as f:
            f.write(f"{params['hidden_dim']},{params['lr']},{params['batch_size']},{params['epochs']},{params['num_gin_layers']},{params['num_lin_layers']},{avg_val_loss},{avg_mae},{test_avg_loss},{test_avg_mae}\n")

    return best_model, best_params

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
    device = torch.device("cuda:0")

    best_model, best_params = grid_search(
        train_dataloader, test_dataloader, input_dim, output_dim, device
    )

    print("Best Model:", best_model)
    print("Best Parameters:", best_params)

if __name__ == "__main__":
    main()
