import json
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from molprop_prediction.scripts.utils import (prompt_user_for_args, 
                                              read_train_data,
                                              read_tabular_train,
                                              preprocess_graph_data)
from molprop_prediction.models.GIN import GIN
from molprop_prediction.models.GAT import GATGraphRegressor
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    model_name, config_path, save_path = prompt_user_for_args()
    device = torch.device("cuda:0")
    train_data = read_train_data()
    with open(config_path, 'r') as file:
        params = json.load(file)
    if model_name == "RF":
        train_data = read_tabular_train()
        X_train, y_train = train_data.drop('y', axis=1), train_data['y']
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        joblib.dump(model, save_path + ".pkl")
        print(f"Model saved to {save_path}")

    if model_name == "GIN":
        # Loading Parameters
        locals().update(params)

        # Loading Model
        model = GIN(dim_h=hidden_dim,
                    num_node_features=input_dim,
                    num_gin_layers=num_gin_layers,
                    num_lin_layers=num_lin_layers,
                    ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        mae = nn.L1Loss()

        # Boucle d'entraînement
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_mae = 0

            for batch in train_dataloader:
                optimizer.zero_grad()
                batch = batch.to(device)  # Move the entire batch to the GPU
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

            average_loss = total_loss / len(train_dataloader)
            average_mae = total_mae / len(train_dataloader)

            print(f"Epoch {epoch + 1}, Loss: {average_loss}, MAE: {average_mae}")

        # Save the trained model and optimizer state
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": average_loss,
                "mae": average_mae,
            },
            save_path,
        )

        print(f"Model saved to {save_path}")

    if model_name == "GAT":
        for epoch in range(hyperparameters['epochs']):
        model.train()
        total_loss = 0
        for data in train_dataloader:
            if data is None:
                continue
            
            optimizer.zero_grad()
            output = model(data)
            output = output.squeeze(-1)
            loss = loss_func(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        