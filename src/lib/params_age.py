params = {
    'device': 'cuda',
    'batch_size': 16,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'n_epochs': 5
}
params_model = {
    'hidden_channels': 100,
    'out_channels': 6,
    'num_blocks': 3,
    'num_bilinear': 100,
    'num_spherical': 3,
    'num_radial': 3,
    'max_num_neighbors': 200,
    'num_before_skip': 2,
    'num_after_skip': 2,
    'num_output_layers': 3,
    'act': 'swish'
}
