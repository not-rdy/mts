params = {
    'device': 'cuda',
    'batch_size': 16,
    'lr': 5e-4,
    'weight_decay': 5e-4,
    'n_epochs': 20
}
params_model = {
    'in_channels': 18,
    'hidden_channels': 100,
    'num_layers': 3,
    'out_channels': 1,
    'dropout': 0.3,
    'act': 'relu',
    'aggr': 'max',
    'jk': 'cat'
}
