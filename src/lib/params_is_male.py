from torch_geometric.nn.norm import BatchNorm

norm = BatchNorm(in_channels=100)

params = {
    'device': 'cuda',
    'batch_size': 32,
    'lr': 5e-4,
    'weight_decay': 5e-5,
    'n_epochs': 20
}
params_model = {
    'in_channels': 18,
    'hidden_channels': 100,
    'num_layers': 3,
    'out_channels': 100,
    'dropout': 0.3,
    'act': 'relu',
    'aggr': 'mean',
    'jk': 'cat',
    'norm': norm
}
