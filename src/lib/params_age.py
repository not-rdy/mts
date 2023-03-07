params = {
    'device': 'cuda',
    'batch_size': 32,
    'lr': 1e-4,
    'weight_decay': 5e-4,
    'n_epochs': 30
}
params_model = {
    'in_channels': 18,
    'hidden_channels': 100,
    'num_layers': 4,
    'out_channels': 6,
    'dropout': 0.2,
    'act': 'relu',
    'aggr': 'max',
    'jk': 'cat'
}
