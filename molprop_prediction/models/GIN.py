import torch
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, global_add_pool

class GIN(torch.nn.Module):
    """Graph isomorphic network (GIN) for graph-level regression."""

    def __init__(self, dim_h, num_node_features, num_gin_layers, num_lin_layers):
        super(GIN, self).__init__()

        self.num_gin_layers = num_gin_layers
        self.num_lin_layers = num_lin_layers

        # Create GINConv layers dynamically based on num_gin_layers
        self.convs = nn.ModuleList([
            GINConv(
                Sequential(
                    Linear(num_node_features if i == 0 else dim_h, dim_h),
                    BatchNorm1d(dim_h),
                    ReLU(),
                    Linear(dim_h, dim_h),
                    ReLU(),
                )
            ) for i in range(num_gin_layers)
        ])

        # Create Linear layers dynamically based on num_lin_layers
        self.lins = nn.ModuleList([
            Linear(dim_h * (i + 1), dim_h * (i + 1)) for i in range(num_lin_layers)
        ])

        self.lin_final = Linear(dim_h * num_lin_layers, 1)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        h = x
        for conv, lin in zip(self.convs, self.lins):
            h = conv(h, edge_index)
            h = global_add_pool(h, batch)
            h = lin(h)
            h = h.relu()

        # Classifier
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin_final(h)

        return F.leaky_relu(h)
