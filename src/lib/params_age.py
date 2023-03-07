params = {
    'device': 'cuda',
    'batch_size': 16,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'n_epochs': 10
}
params_model = {
    'in_channels': 18,
    'hidden_channels': 100,
    'num_layers': 3,
    'out_channels': 100,
    'dropout': 0.3,
    'act': 'relu',
    'aggr': 'max',
    'jk': 'lstm'
}
params_agg_lstm = {
    'in_channels': 100,
    'out_channels': 6,
    'num_layers': 2,
    'dropout': 0.3,
    'bidirectional': False
}
