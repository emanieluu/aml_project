import argparse
import json
import torch
from torch_geometric.loader import DataLoader
from molprop_prediction.models.GIN_bis import GIN
from molprop_prediction.scripts.preprocess_bis import graph_datalist_from_smiles_and_labels
import pandas as pd
import torch.optim as optim

device = torch.device("cuda:0")

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement du modèle GIN")
    args = parser.parse_args()
    args.config = "molprop_prediction/configs/params.json"
    return args

def load_params(config_path):
    with open(config_path, "r") as config_file:
        params = json.load(config_file)
    return params

# Function to load the trained model
def load_model(model, optimizer, checkpoint_path):
    loaded_checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(loaded_checkpoint['model_state_dict'])
    optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    epoch = loaded_checkpoint['epoch']
    loss = loaded_checkpoint['loss']
    mae = loaded_checkpoint['mae']
    print(f"Model loaded from {checkpoint_path}, trained for {epoch} epochs. Last loss: {loss}, Last MAE: {mae}")
    return model, optimizer

# Paramètres du modèle et de l'entraînement
args = parse_args()
params = load_params(args.config)
input_dim = 79
hidden_dim = params["hidden_dim"]
output_dim = params["output_dim"]
lr = params["lr"]
epochs = params["epochs"]
batch_size = params["batch_size"]


# Load data for prediction (replace with your new data)
new_data = pd.read_csv("./data/raw_data/X_test.csv")
new_dataset = graph_datalist_from_smiles_and_labels(new_data["smiles"], new_data["y"])
new_dataloader = DataLoader(new_dataset, batch_size=16, shuffle=False)

# Load the model and optimizer
model = GIN(dim_h=params["hidden_dim"],
            num_node_features=input_dim,
            num_gin_layers=params["num_gin_layers"],
            num_lin_layers=params["num_lin_layers"],
            ).to(device)  # Initialize the model with the same architecture
optimizer = optim.Adam(model.parameters(), lr=lr)  # You can adjust lr if needed
checkpoint_path = "/home/onyxia/work/aml_project/molprop_prediction/models/new_saved_models"  # Update with the path where you saved the model
model, optimizer = load_model(model, optimizer, checkpoint_path)

# Set the model to evaluation mode
model.eval()

# Make predictions on new data
predictions = []
with torch.no_grad():
    for batch in new_dataloader:
        batch = batch.to(device)
        x, edge_index, batch_data = batch.x, batch.edge_index, batch.batch
        output = model(x, edge_index, batch_data)
        predictions.extend(output.cpu().numpy().flatten().tolist())

# Create a DataFrame with 'id' and 'prediction' columns
predictions_df = pd.DataFrame({'id': new_data['id'], 'y': predictions})

# Save the DataFrame to a CSV file
predictions_csv_path = '/home/onyxia/work/aml_project/data/raw_data/predictions/new_predictions.csv'
predictions_df.to_csv(predictions_csv_path, index=False)

print(f'Predictions saved to {predictions_csv_path}')
