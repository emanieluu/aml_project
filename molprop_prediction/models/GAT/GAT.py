import torch
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, global_add_pool

class GATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GATLayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.att = torch.nn.Parameter(torch.Tensor(1, 2 * out_channels))
        self.leakyrelu = torch.nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.att)
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linear transformation
        x = self.lin(x)

        # Start propagating messages.
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index, size):
        # Compute attention coefficients.
        x = torch.cat([x_i, x_j], dim=-1)
        alpha = self.leakyrelu(torch.matmul(x, self.att.t()))
        alpha = F.softmax(alpha, dim=1)

        return x_j * alpha

    def aggregate(self, inputs, index, dim_size=None):
        # The aggregation step simply sums up the transformed messages.
        return torch.sum(inputs, dim=0)

class GATGraphRegressor(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, out_features):
        super(GATGraphRegressor, self).__init__()
        #self.layers.append(GATLayer(num_node_features, hidden_dims[0]))

        
        self.conv1 = GATLayer(num_node_features, hidden_dim, heads=8, dropout=0.6)
        self.conv2 = GATLayer(hidden_dim * 8, hidden_dim, heads=1, dropout=0.6)
        self.fc = torch.nn.Linear(hidden_dim, out_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))

        # Agrégation globale pour obtenir une représentation unique du graphe
        x = global_mean_pool(x, batch)

        # Couche linéaire pour la prédiction de la valeur de régression (pIC50)
        x = self.fc(x)

        return x
