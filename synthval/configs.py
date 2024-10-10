import torch
import pynever.strategies.training
import pynever.nodes

DEFAULT_NETWORK_PARAMS = {
    "network_id":               "MetricNetwork",
    "num_hidden_neurons":       [256, 128, 64]
}

DEFAULT_TRAINING_PARAMS = {
    "optimizer_con":            torch.optim.Adam,
    "opt_params":               {"lr": 0.01},
    "n_epochs":                 10,
    "validation_percentage":    0.3,
    "train_batch_size":         128,
    "validation_batch_size":    64,
    "r_split":                  True,
    "scheduler_con":            None,
    "sch_params":               None,
    "precision_metric":         pynever.strategies.training.PytorchMetrics.inaccuracy,
    "network_transform":        None,
    "device":                   "cpu",
    "train_patience":           10,
    "checkpoints_root":         "output/classifiers/checkpoints/",
    "verbose_rate":             10
}

DEFAULT_TESTING_PARAMS = {
    "metric":                   pynever.strategies.training.PytorchMetrics.inaccuracy,
    "metric_params":            {},
    "test_batch_size":          1,
    "device":                   "cpu"
}