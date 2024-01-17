import json
import joblib
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from molprop_prediction.models.GIN import GIN
from molprop_prediction.scripts.utils import (prompt_user_for_predictions,
                                              read_test_data,
                                              read_tabular_test,
                                              load_model,
                                              preprocess_graph_data)

if __name__ == "__main__":
    model, checkpoint_path, config_path, save_path = prompt_user_for_predictions()
    device = torch.device("cuda:0")
    test_data = read_test_data()
    with open(config_path, 'r') as file:
        params = json.load(file)
    if model == "RF":
        test_data = read_tabular_test()
        X_test, y_test = test_data.drop("y", axis=1), test_data['y']
        model = joblib.load(checkpoint_path)
        predictions = model.predict(X_test)
        predictions_df = pd.DataFrame({'predictions': predictions})
        predictions_df.to_csv(save_path, index=False)
        mae = mean_absolute_error(y_test, predictions)
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'Predictions saved to {save_path}')
    if model == "GIN":
        #Loading Data
        test_dataloader = preprocess_graph_data(test_data)
        #test_dataloader, kept_test_id = load_graph_preprocessed_test_dataset()
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
