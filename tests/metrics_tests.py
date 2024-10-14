import pandas

import synthval.metrics
import synthval.features_extraction
import torch
import pynever.strategies.training


def test_kl_divergence():

    ori_features_df = pandas.read_csv("test_data/test_ori_features.csv", header=None)
    gen_features_df = pandas.read_csv("test_data/test_gen_features.csv", header=None)
    metric = synthval.metrics.KLDivergenceEstimation()
    metric.calculate(ori_features_df, gen_features_df)


def test_wasserstein_distance():

    ori_features_df = pandas.read_csv("test_data/test_ori_features.csv", header=None)
    gen_features_df = pandas.read_csv("test_data/test_gen_features.csv", header=None)
    metric = synthval.metrics.WassersteinDistance()
    metric.calculate(ori_features_df, gen_features_df)


def test_energy_distance():

    ori_features_df = pandas.read_csv("test_data/test_ori_features.csv", header=None)
    gen_features_df = pandas.read_csv("test_data/test_gen_features.csv", header=None)
    metric = synthval.metrics.EnergyDistance()
    metric.calculate(ori_features_df, gen_features_df)


def test_mahalanobis_distance():

    ori_features_df = pandas.read_csv("test_data/test_ori_features.csv", header=None)
    gen_features_df = pandas.read_csv("test_data/test_gen_features.csv", header=None)
    metric = synthval.metrics.MeanMahalanobisDistance()
    metric.calculate(ori_features_df, gen_features_df)


def test_fcnn_accuracy():

    ori_features_df = pandas.read_csv("test_data/test_ori_features.csv", header=None)
    gen_features_df = pandas.read_csv("test_data/test_gen_features.csv", header=None)

    test_training_params = {
        "optimizer_con": torch.optim.Adam,
        "opt_params": {"lr": 0.01},
        "n_epochs": 5,
        "validation_percentage": 0.3,
        "train_batch_size": 8,
        "validation_batch_size": 2,
        "r_split": True,
        "scheduler_con": None,
        "sch_params": None,
        "precision_metric": pynever.strategies.training.PytorchMetrics.inaccuracy,
        "network_transform": None,
        "device": "cpu",
        "train_patience": 5,
        "checkpoints_root": "test_data/checkpoints/",
        "verbose_rate": 1
    }

    test_network_params = {
        'network_id': 'TestMetricNetwork',
        "num_hidden_neurons": [256, 128, 64]
    }

    metric = synthval.metrics.FCNNAccuracyMetric(training_params=test_training_params,
                                                 network_params=test_network_params)

    metric.calculate(ori_features_df, gen_features_df)


def test_inception_score():

    probabilities_df = pandas.read_csv("test_data/test_probabilities.csv", header=None)
    metric = synthval.metrics.InceptionScore()
    metric.calculate(probabilities_df)


if __name__ == '__main__':
    test_kl_divergence()
    test_wasserstein_distance()
    test_energy_distance()
    test_mahalanobis_distance()
    test_fcnn_accuracy()
    test_inception_score()
