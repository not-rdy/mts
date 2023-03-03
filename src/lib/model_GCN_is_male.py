import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import MaxAggregation

params_GCN = {
    'device': 'cuda',
    'batch_size': 16,
    'lr': 0.01,
    'weight_decay': 5e-4,
    'n_epochs': 1,
    'threshold': 0.5
}


class GCN_is_male(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = GCNConv(19, 500)
        self.conv2 = GCNConv(500, 400)
        self.conv3 = GCNConv(400, 300)
        self.conv4 = GCNConv(300, 200)
        self.conv5 = GCNConv(200, 100)
        self.agg = MaxAggregation()
        self.lin = torch.nn.Linear(100, 1)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.agg(x, index=graph.batch)
        x = self.lin(x)
        out = torch.sigmoid(x)
        return out
