params = {
    'device': 'cuda',
    'batch_size': 64,
    'lr': 5e-5,
    'weight_decay': 5e-5,
    'n_epochs': 20,
    'weights_loss': [0.151, 0.0152, 0.079, 0.134, 0.27, 0.3508]
}
params_model = {
    'in_channels': 18,
    'hidden_channels': 100,
    'num_layers': 25,
    'out_channels': 100,
    'dropout': 0.3,
    'act': 'relu',
    'aggr': 'max',
    'jk': 'cat'
}
