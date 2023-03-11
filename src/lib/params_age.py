params = {
    'device': 'cuda',
    'batch_size': 64,
    'lr': 5e-4,
    'weight_decay': 5e-5,
    'n_epochs': 100,
    'weights_loss': [0.151, 0.0152, 0.079, 0.134, 0.27, 0.3508]
}
params_model = {
    'in_channels': 18,
    'hidden_channels': 100,
    'num_layers': 6,
    'out_channels': 6,
    'dropout': 0.3,
    'act': 'relu',
    'aggr': 'max',
    'jk': 'cat'
}
