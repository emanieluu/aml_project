import json 
import pandas as pd
import torch
import torch.optim as optim
from molprop_prediction.scripts.utils import (prompt_user_for_predictions, load_graph_preprocessed_test_dataset, load_model)
from molprop_prediction.models.GIN import GIN

if __name__ == "__main__":
    model, checkpoint_name, params_file = prompt_user_for_predictions()
    device = torch.device("cuda:0")
    print(f"Using {model} model with parameters from the file {params_file} and checkpoint {checkpoint_name} to predict")
    checkpoint_path = "./molprop_prediction/models/trained_models/" + checkpoint_name
    config_path = "./molprop_prediction/configs/" + params_file
    save_path = "./data/predictions/" + checkpoint_name + "_predictions.csv"

    with open(config_path, 'r') as file:
        params = json.load(file)
    if model == "RF":
        pass
    if model == "GIN":
        #Loading Data
        test_dataloader, kept_test_id = load_graph_preprocessed_test_dataset()
        #Loading Parameters
        locals().update(params)
        #Loading Model and 
        model = GIN(dim_h=hidden_dim,
            num_node_features=input_dim,
            num_gin_layers=num_gin_layers,
            num_lin_layers=num_lin_layers,
            ).to(device)  # Initialize the model with the same architecture
        optimizer = optim.Adam(model.parameters(), lr=lr)  # You can adjust lr if needed
        model, optimizer = load_model(model, optimizer, checkpoint_path)
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_dataloader:
                batch = batch.to(device)
                x, edge_index, batch_data = batch.x, batch.edge_index, batch.batch
                output = model(x, edge_index, batch_data)
                predictions.extend(output.cpu().numpy().flatten().tolist())

        # Create a DataFrame with 'id' and 'prediction' columns
        predictions_df = pd.DataFrame({'id': kept_test_id, 'y': predictions})

        # Save the DataFrame to a CSV file
        predictions_df.to_csv(save_path, index=False)

        print(f'Predictions saved to {save_path}')

    if model == "GAT":
        pass
