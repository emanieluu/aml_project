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
        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, 1)

    def forward(self, x, edge_index, batch):
        # Node embeddings

        h = [torch.zeros_like(x)] * (self.num_gin_layers + 1)
        h[0] = x
        for i, conv in enumerate(self.convs):
            h[i + 1] = conv(h[i], edge_index)

        # Graph-level readout
        for i in range(1, self.num_gin_layers + 1):
            h[i] = global_add_pool(h[i], batch)
        
        # Concatenate graph embeddings
        h = torch.cat([h[i] for i in range(1, self.num_gin_layers + 1)], dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return F.leaky_relu(h)

