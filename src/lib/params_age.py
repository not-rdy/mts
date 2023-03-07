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
    'out_channels': 6,
    'depth': 3,
    'dropout': 0.2,
    'act': 'relu'
}
