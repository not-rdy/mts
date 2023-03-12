params = {
    'device': 'cuda',
    'batch_size': 16,
    'lr': 5e-5,
    'weight_decay': 5e-5,
    'n_epochs': 100
}
params_model = {
    'in_channels': 18,
    'hidden_channels': 100,
    'num_layers': 10,
    'out_channels': 100,
    'dropout': 0.3,
    'act': 'relu',
    'aggr': 'mean',
    'jk': 'cat'
}
