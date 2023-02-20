import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import MulAggregation

params_GCN = {
    'device': 'cuda',
    'batch_size': 32,
    'lr': 0.01,
    'weight_decay': 5e-4,
    'n_epochs': 20
}


class GCN(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = torch.nn.Linear(19, 100)
        self.conv1 = GCNConv(100, 100)
        self.conv2 = GCNConv(100, 40)
        self.lin2 = torch.nn.Linear(40, 6)
        self.agg = MulAggregation()

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.lin1(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = self.agg(x, graph.batch)
        out = F.softmax(x, dim=0)
        return out
