"""
Module for defining default parameters for network training and testing using PyTorch and pyNeVer.

This module provides default configurations for setting up neural networks and training them
using the PyTorch framework, with an emphasis on integrating with pyNeVer strategies.

Attributes
----------
DEFAULT_NETWORK_PARAMS : dict
    Default parameters for the neural network architecture. The dictionary contains:

    - network_id : str
        Identifier for the network architecture.
    - num_hidden_neurons : list of int
        Number of neurons for each hidden layer in the network.

DEFAULT_TRAINING_PARAMS : dict
    Default parameters for training the network using PyTorch. The dictionary contains:

    - optimizer_con : torch.optim.Optimizer
        Constructor for the optimizer to be used during training.
    - opt_params : dict
        Parameters for the optimizer, such as learning rate (`lr`).
    - n_epochs : int
        Number of epochs for training.
    - validation_percentage : float
        Proportion of the data used for validation.
    - train_batch_size : int
        Batch size used for training.
    - validation_batch_size : int
        Batch size used for validation.
    - r_split : bool
        Whether to perform a random data split for training/validation.
    - scheduler_con : torch.optim.lr_scheduler, optional
        Constructor for a learning rate scheduler (default is None).
    - sch_params : dict or None
        Parameters for the learning rate scheduler (default is None).
    - precision_metric : pynever.strategies.training.PytorchMetrics
        Metric to evaluate training precision, e.g., accuracy or inaccuracy.
    - network_transform : callable, optional
        Transformation function to apply to the network (default is None).
    - device : str
        The device on which the training will be performed (e.g., 'cpu' or 'cuda').
    - train_patience : int
        Number of epochs to wait before early stopping if no improvement in validation loss.
    - checkpoints_root : str
        Path to the directory where model checkpoints will be saved.
    - verbose_rate : int
        Frequency (in epochs) for printing training progress.

DEFAULT_TESTING_PARAMS : dict
    Default parameters for testing the trained network. The dictionary contains:

    - metric : pynever.strategies.training.PytorchMetrics
        Metric used to evaluate the model performance on the test set.
    - metric_params : dict
        Additional parameters for the testing metric.
    - test_batch_size : int
        Batch size for testing.
    - device : str
        The device on which testing will be performed (e.g., 'cpu' or 'cuda').

"""

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