params = {
    'device': 'cuda',
    'batch_size': 64,
    'lr': 5e-6,
    'weight_decay': 5e-6,
    'n_epochs': 100
}
params_model = {
    'in_channels': 18,
    'hidden_channels': 100,
    'num_layers': 4,
    'out_channels': 1,
    'dropout': 0.2,
    'act': 'relu',
    'aggr': 'max',
    'jk': 'cat'
}
