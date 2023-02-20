import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import MaxAggregation

params_GCN = {
    'device': 'cuda',
    'batch_size': 4,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'n_epochs': 100
}


class GCN(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = torch.nn.Linear(19, 100)
        self.conv1 = GCNConv(100, 90)
        self.conv2 = GCNConv(90, 70)
        self.conv3 = GCNConv(70, 50)
        self.agg = MaxAggregation()
        self.lin2 = torch.nn.Linear(50, 40)
        self.lin3 = torch.nn.Linear(40, 30)
        self.lin4 = torch.nn.Linear(30, 20)
        self.lin5 = torch.nn.Linear(20, 10)
        self.lin6 = torch.nn.Linear(10, 6)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.lin1(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.agg(x, graph.batch)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin4(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin5(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin6(x)
        out = F.softmax(x, dim=0)
        return out
