import json
import joblib
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from molprop_prediction.models.GIN_bis import GIN
from molprop_prediction.scripts.utils import (
    prompt_user_for_predictions,
    read_test_data,
    read_train_data,
    read_tabular_test,
    load_model,
    preprocess_graph_data,
    preprocess_test_graph_data,
    load_data_gat,
)
from molprop_prediction.models.GAT import GATGraphRegressor
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    (
        model,
        checkpoint_path,
        config_path,
        save_path,
    ) = prompt_user_for_predictions()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(config_path, "r") as file:
        params = json.load(file)

    if model == "RF":
        test_data = read_tabular_test()
        X_test, y_test = test_data.drop("y", axis=1), test_data["y"]
        model = joblib.load(checkpoint_path)
        predictions = model.predict(X_test)

        predictions_df = pd.DataFrame({"predictions": predictions})
        predictions_df.to_csv(save_path, index=False)

        mae = mean_absolute_error(y_test, predictions)
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Predictions saved to {save_path}")

    if model == "GIN":
        test_data = read_test_data()
        X_test, y_test = test_data.drop("y", axis=1), test_data["y"]
        # Loading Parameters
        input_dim = params["input_dim"]
        hidden_dim = params["hidden_dim"]
        output_dim = params["output_dim"]
        lr = params["lr"]
        epochs = params["epochs"]
        batch_size = params["batch_size"]
        num_gin_layers = params["num_gin_layers"]
        num_lin_layers = params["num_lin_layers"]

        # Loading Data
        test_dataloader = preprocess_test_graph_data(test_data)

        # Loading Model
        model = GIN(hidden_dim, input_dim).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        mae = nn.L1Loss()
        model, optimizer = load_model(model, optimizer, checkpoint_path)
        model.eval()

        predictions = []
        # total_mae = 0

        with torch.no_grad():
            for batch in test_dataloader:
                batch = batch.to(device)
                x, edge_index, batch_data = (
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                )
                output = model(x, edge_index, batch_data)
                predictions.extend(output.cpu().numpy().flatten().tolist())
                # mae_value = mae(output, batch.y.view(-1, 1))
                # total_mae += mae_value.item()

        mae = mean_absolute_error(y_test, predictions)
        # average_mae = total_mae / len(test_dataloader)
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Predictions saved to {save_path}")

        predictions_df = pd.DataFrame(
            {"id": test_data.index, "y": predictions}
        )
        predictions_df.to_csv(save_path, index=False)

    if model == "GAT":
        test_data = read_test_data()
        y_test = test_data["y"]

        # Loading Parameters
        num_node_features = params["num_node_features"]
        hidden_dim = params["hidden_dim"]
        out_features = params["out_features"]
        epochs = params["epochs"]
        lr = params["lr"]

        # Loading Data
        test_dataloader = load_data_gat(test_data)

        # Loading Model
        model = GATGraphRegressor(
            num_node_features, hidden_dim, out_features
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        mae = nn.L1Loss()
        model, optimizer = load_model(model, optimizer, checkpoint_path)
        model.eval()

        predictions = []
        total_mae = 0

        with torch.no_grad():
            for batch in test_dataloader:
                batch = batch.to(device)
                output = model(batch)
                predictions.extend(output.cpu().numpy().flatten().tolist())
                mae_value = mae(output, batch.y.view(-1, 1))
                total_mae += mae_value.item()
        average_mae = total_mae / len(test_dataloader)
        print(f"Mean Absolute Error (MAE): {average_mae}")
        print(f"Predictions saved to {save_path}")

        predictions_df = pd.DataFrame(
            {"id": test_data.index, "y": predictions}
        )
        predictions_df.to_csv(save_path, index=False)
