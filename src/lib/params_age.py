params = {
    'device': 'cuda',
    'batch_size': 16,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'n_epochs': 5
}
params_model = {
    'in_channels': 18,
    'hidden_channels': 100,
    'num_layers': 3,
    'out_channels': 100,
    'dropout': 0.2,
    'act': 'relu',
    'aggr': 'max',
    'jk': 'lstm'
}
params_agg_lstm = {
    'in_channels': 100,
    'out_channels': 6,
    'hidden_size': 100,
    'num_layers': 1,
    'batch_first': False,
    'dropout': 0.2,
    'bidirectional': False
}
