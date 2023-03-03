import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import MaxAggregation

params_GCN = {
    'device': 'cuda',
    'batch_size': 32,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'n_epochs': 50
}


class GCN_age(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.lin0 = torch.nn.Linear(19, 500)
        self.conv1 = GCNConv(500, 500)
        self.lin1 = torch.nn.Linear(500, 300)
        self.lin2 = torch.nn.Linear(300, 100)
        self.lin3 = torch.nn.Linear(100, 6)
        self.agg = MaxAggregation()

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.lin0(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin3(x)
        x = self.agg(x, index=graph.batch)
        out = F.softmax(x, dim=0)
        return out
