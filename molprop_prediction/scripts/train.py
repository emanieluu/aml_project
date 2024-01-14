import json 
import torch
import torch.nn as nn
import torch.optim as optim
from molprop_prediction.scripts.utils import prompt_user_for_args, load_graph_preprocessed_dataset
from molprop_prediction.models.GIN import GIN

if __name__ == "__main__":
    model, params_file, model_name = prompt_user_for_args()
    device = torch.device("cuda:0")
    print(f"Training the {model} model with parameters from the file {params_file} and saving it as {model_name}.")
    config_path = "./molprop_prediction/configs/" + params_file
    save_path = "./molprop_prediction/models/trained_models/" + model_name
    with open(config_path, 'r') as file:
        params = json.load(file)
    if model == "RF":
        pass
    if model == "GIN":
        #Loading Data
        train_dataloader, test_dataloader = load_graph_preprocessed_dataset()

        #Loading Parameters
        locals().update(params)
        #Loading Model and 
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

    if model == "GAT":
        pass
