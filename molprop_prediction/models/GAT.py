import torch
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F



class GATGraphRegressor(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, out_features):
        super(GATGraphRegressor, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv4 = GATConv(hidden_dim, hidden_dim)
        self.bn4 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = global_mean_pool(x, data.batch)
        return self.fc(x)

