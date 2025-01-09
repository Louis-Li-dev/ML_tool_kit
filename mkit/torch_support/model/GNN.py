import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNLayerWithSkip(nn.Module):
    """
    A GCN layer with skip connections.

    Args:
        in_channels (int): Input feature size.
        out_channels (int): Output feature size.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(GCNLayerWithSkip, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        Forward pass with skip connection.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Graph edge index.

        Returns:
            torch.Tensor: Node features after GCN and skip connection.
        """
        x_gcn = self.gcn(x, edge_index)
        x_skip = self.linear(x)
        return F.mish(x_gcn + x_skip)

class GNNRegressionWithSkipConnections(nn.Module):
    """
    A GNN for regression with skip connections and flexible layers.

    Args:
        num_node_features (int): Number of node features.
        hidden_channels (int): Size of hidden layers.
        out_channels (int): Size of output features.
        num_layers (int): Number of GCN layers.
    """
    def __init__(self, num_node_features: int, hidden_channels: int, out_channels: int, num_layers: int):
        super(GNNRegressionWithSkipConnections, self).__init__()
        
        # Preprocessing layers
        self.preprocess = nn.Sequential(
            nn.Linear(num_node_features, hidden_channels),
            nn.Mish()
        )
        
        # Dynamically create GCN layers with skip connections
        self.gcn_layers = nn.ModuleList([
            GCNLayerWithSkip(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])
        
        # Postprocessing layers
        self.postprocess = nn.Sequential(
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, data):
        """
        Forward pass through the model.

        Args:
            data (torch_geometric.data.Data): Input graph data.

        Returns:
            torch.Tensor: Predicted output features.
        """
        x, edge_index = data.x, data.edge_index

        x = self.preprocess(x)
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)
        x = self.postprocess(x)
        return x
