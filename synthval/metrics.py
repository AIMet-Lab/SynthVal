"""
Module for computing various similarity metrics between two sets of samples originating from multivariate distributions.

This module defines abstract and concrete classes for computing similarity metrics between samples from two
distributions. The available metrics include Kullback-Leibler divergence, Wasserstein distance, Energy distance,
Mean Mahalanobis distance, and a machine learning-based accuracy metric using fully connected neural networks.

Classes
-------
SimilarityMetric(abc.ABC)
    Abstract base class for defining similarity metrics.
KLDivergenceEstimation(SimilarityMetric)
    Concrete implementation of Kullback-Leibler divergence estimation.
WassersteinDistance(SimilarityMetric)
    Concrete implementation of the Wasserstein distance.
EnergyDistance(SimilarityMetric)
    Concrete implementation of the Energy distance.
MeanMahalanobisDistance(SimilarityMetric)
    Concrete implementation of the mean Mahalanobis distance.
FCNNAccuracyMetric(SimilarityMetric)
    Concrete implementation of an accuracy metric based on fully-connected neural networks.

"""

import abc

import numpy as np
import scipy.spatial as sci_sp
import scipy.stats as sci_stats
import dcor
import pandas

import synthval.utilities
import pynever.strategies.training
import pynever.networks
import pynever.nodes
import sklearn.utils
import torch
import synthval.configs
import typing


class SimilarityMetric(abc.ABC):
    """
    Abstract base class representing a generic similarity metric between two sets of samples originating from
    two multivariate distributions.
    Child classes must implement the concrete `calculate` method for computing the specific metric.

    """

    @abc.abstractmethod
    def calculate(self, dist_p_df: pandas.DataFrame, dist_q_df: pandas.DataFrame) -> float:
        """
        Abstract method to compute a metric of similarity between two set of samples originating from two multivariate
        distribution P and Q.

        Parameters
        ----------
        dist_p_df : pandas.DataFrame
            Set of samples representing distribution P.
        dist_q_df : pandas.DataFrame
            Set of samples representing distribution Q.

        Returns
        -------
        float
            The value of the metric.
        """

        raise NotImplementedError


class KLDivergenceEstimation(SimilarityMetric):
    """
    Similarity Metric computing an estimation of the Kullback-Leibler divergence based on the methodology proposed in
    the referenced paper.

    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of continuous distributions IEEE International Symposium
    on Information Theory, 2008.

    """

    def __init__(self):
        super().__init__()

    def calculate(self, dist_p_df: pandas.DataFrame, dist_q_df: pandas.DataFrame) -> float:
        """
        Compute an estimation of the Kullback-Leibler divergence between two set of samples originating from two
        multivariate distribution P and Q.

        Parameters
        ----------
        dist_p_df : pandas.DataFrame
            Set of samples representing distribution P.
        dist_q_df : pandas.DataFrame
            Set of samples representing distribution Q.

        Returns
        -------
        float
            The estimated value of the Kullback-Leibler divergence.
        """

        dist_p = dist_p_df.values
        dist_q = dist_q_df.values

        n, d = dist_p.shape
        m, d_s = dist_q.shape

        assert d == d_s

        # Build a KD tree representation of the samples and find the nearest neighbour
        # of each point in first_dist.

        p_tree = sci_sp.cKDTree(dist_p)
        q_tree = sci_sp.cKDTree(dist_q)

        # Get the first two nearest neighbours for p_dist, since the closest one is the
        # sample itself.
        r = p_tree.query(dist_p, k=2, eps=.01, p=2)[0][:, 1]
        s = q_tree.query(dist_p, k=1, eps=.01, p=2)[0]

        # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
        # on the first term of the right hand side.
        return np.log(s / r).sum() * d / n + np.log(m / (n - 1.0))


class WassersteinDistance(SimilarityMetric):
    """
    Similarity Metric computing the Wasserstein Distance.

    """

    def __init__(self):
        super().__init__()

    def calculate(self, dist_p_df: pandas.DataFrame, dist_q_df: pandas.DataFrame) -> float:
        """
        Compute the Wasserstein Distance between two set of samples originating from two
        multivariate distribution P and Q.

        Parameters
        ----------
        dist_p_df : pandas.DataFrame
            Set of samples representing distribution P.
        dist_q_df : pandas.DataFrame
            Set of samples representing distribution Q.

        Returns
        -------
        float
            The value of the Wasserstein Distance.
        """

        dist_p = dist_p_df.values
        dist_q = dist_q_df.values
        return sci_stats.wasserstein_distance_nd(dist_p, dist_q)


class EnergyDistance(SimilarityMetric):
    """
    Similarity Metric computing the Energy Distance.

    """

    def __init__(self):
        super().__init__()

    def calculate(self, dist_p_df: pandas.DataFrame, dist_q_df: pandas.DataFrame) -> float:
        """
        Compute the Energy Distance between two set of samples originating from two
        multivariate distribution P and Q.

        Parameters
        ----------
        dist_p_df : pandas.DataFrame
            Set of samples representing distribution P.
        dist_q_df : pandas.DataFrame
            Set of samples representing distribution Q.

        Returns
        -------
        float
            The value of the Wasserstein Distance.
        """

        dist_p = dist_p_df.values
        dist_q = dist_q_df.values
        return dcor.energy_distance(dist_p, dist_q)


class MeanMahalanobisDistance(SimilarityMetric):
    """
    Similarity Metric computing the mean of the estimated Mahalanobis Distances between all the samples
    of the P distribution and the Q distribution (the estimation is due to the use of the numpy.cov method
    to compute the covariance matrix of the Q distribution).

    """

    def __init__(self):
        super().__init__()

    def calculate(self, dist_p_df: pandas.DataFrame, dist_q_df: pandas.DataFrame) -> float:
        """
        Compute an estimation of the Mahalanobis Distance between two distributions as a mean of
        the Mahalanobis Distance computed over each sample of the first against the second.

        Parameters
        ----------
        dist_p_df : pandas.DataFrame
            Set of samples representing distribution P.
        dist_q_df : pandas.DataFrame
            Set of samples representing distribution Q.

        Returns
        -------
        out : float
        The estimated mean Mahalanobis Distance.

        """

        dist_p = dist_p_df.values
        dist_q = dist_q_df.values

        covariance_matrix = np.cov(dist_q.T)
        mean_vector = np.mean(dist_q, axis=0)

        m_distances = []
        for i in range(dist_p.shape[0]):
            m_distances.append(sci_sp.distance.mahalanobis(dist_p[i, :], mean_vector, covariance_matrix))

        m_distances = np.array(m_distances)
        return np.mean(m_distances)


class FCNNAccuracyMetric(SimilarityMetric):
    """
    Similarity Metric computing the Accuracy of a fully-connected neural networks trained to distinguish between the
    points belonging to the distributions P and Q.

    Attributes
    ----------
    test_percentage: float, Optional
        Percentage of the samples to use for the testing of the network, and therefore for computing the
        final metric (default: 0.2).
    rng_seed: int, Optional
        Random Generator seed used for numpy utilities (default: 0).
    network_params: dict, Optional
        Contains the relevant parameters needed to build the network. Refer to configs.DEFAULT_NETWORK_PARAMS for
        an example (default: configs.DEFAULT_NETWORK_PARAMS).
    training_params: dict, Optional
        Contains the relevant parameters needed to train the network. Refer to configs.DEFAULT_TRAINING_PARAMS for
        an example (default: configs.DEFAULT_TRAINING_PARAMS).
    testing_params: dict, Optional
        Contains the relevant parameters needed to test the network. Refer to configs.DEFAULT_TESTING_PARAMS for
        an example (default: configs.DEFAULT_TESTING_PARAMS).

    """

    def __init__(self, test_percentage: float = 0.2, rng_seed: int = 0,
                 network_params: dict = synthval.configs.DEFAULT_NETWORK_PARAMS,
                 training_params: dict = synthval.configs.DEFAULT_TRAINING_PARAMS,
                 testing_params: dict = synthval.configs.DEFAULT_TESTING_PARAMS):
        self.test_percentage = test_percentage
        self.rng_seed = rng_seed
        self.network_params = network_params
        self.training_params = training_params
        self.testing_params = testing_params

        super().__init__()

    def __build_metric_network(self, input_dim, output_dim) -> pynever.networks.SequentialNetwork:
        num_hidden_neurons = self.network_params['num_hidden_neurons']
        network_id = self.network_params['network_id']
        pyn_net = pynever.networks.SequentialNetwork(network_id, "X")
        current_dim = input_dim
        for i in range(len(num_hidden_neurons)):
            hn_num = num_hidden_neurons[i]
            fc_node = pynever.nodes.FullyConnectedNode(f"FC_{i}", current_dim, hn_num)
            pyn_net.add_node(fc_node)
            current_dim = (hn_num,)

            relu_node = pynever.nodes.ReLUNode(f"ReLU_{i}", current_dim)
            pyn_net.add_node(relu_node)

        output_node = pynever.nodes.FullyConnectedNode(f"FC_out", current_dim, output_dim[0])
        pyn_net.add_node(output_node)

        return pyn_net

    def __train_metric_network(self, net, dataset: synthval.utilities.FeaturesDataset) -> \
            pynever.networks.SequentialNetwork:
        train_params = self.training_params
        optimizer_con = train_params["optimizer_con"]
        opt_params = train_params["opt_params"]
        labels = dataset.train_df["Label"].to_numpy(int)
        c_weights = sklearn.utils.class_weight.compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        c_weights = np.float32(c_weights)
        loss_function = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(c_weights))
        n_epochs = train_params["n_epochs"]
        validation_percentage = train_params["validation_percentage"]
        train_batch_size = train_params["train_batch_size"]
        validation_batch_size = train_params["validation_batch_size"]
        r_split = train_params["r_split"]
        scheduler_con = train_params["scheduler_con"]
        sch_params = train_params["sch_params"]
        precision_metric = train_params["precision_metric"]
        network_transform = train_params["network_transform"]
        device = train_params["device"]
        train_patience = train_params["train_patience"]
        checkpoints_root = train_params["checkpoints_root"]
        verbose_rate = train_params["verbose_rate"]

        trainer = pynever.strategies.training.PytorchTraining(optimizer_con=optimizer_con,
                                                              opt_params=opt_params,
                                                              loss_function=loss_function,
                                                              n_epochs=n_epochs,
                                                              validation_percentage=validation_percentage,
                                                              train_batch_size=train_batch_size,
                                                              validation_batch_size=validation_batch_size,
                                                              r_split=r_split,
                                                              scheduler_con=scheduler_con,
                                                              sch_params=sch_params,
                                                              precision_metric=precision_metric,
                                                              network_transform=network_transform,
                                                              device=device,
                                                              train_patience=train_patience,
                                                              checkpoints_root=checkpoints_root,
                                                              verbose_rate=verbose_rate)

        trained_net = trainer.train(net, dataset)
        return trained_net

    def __test_metric_network(self, net, dataset: synthval.utilities.FeaturesDataset) -> float:
        test_params = self.testing_params
        metric = test_params["metric"]
        metric_params = test_params["metric_params"]
        test_batch_size = test_params["test_batch_size"]
        device = test_params["device"]

        tester = pynever.strategies.training.PytorchTesting(metric=metric,
                                                            metric_params=metric_params,
                                                            test_batch_size=test_batch_size,
                                                            device=device,
                                                            save_results=True)

        dataset.train_mode = False
        test_loss = tester.test(net, dataset)
        return test_loss

    def calculate(self, dist_p_df: pandas.DataFrame, dist_q_df: pandas.DataFrame) -> float:
        """
        Compute the Accuracy of a fully-connected neural networks trained to distinguish between the
        points belonging to the distributions P and Q.

        Parameters
        ----------
        dist_p_df : pandas.DataFrame
            Set of samples representing distribution P.
        dist_q_df : pandas.DataFrame
            Set of samples representing distribution Q.

        Returns
        -------
        out : float
        The final accuracy computed on the test set.

        """

        dataset = synthval.utilities.FeaturesDataset(dist_p_df, dist_q_df, self.test_percentage, self.rng_seed)
        input_dim = dataset.__getitem__(0)[0].shape
        output_dim = (2,)

        net = self.__build_metric_network(input_dim, output_dim)
        trained_net = self.__train_metric_network(net, dataset)

        dataset.train_mode = False
        final_accuracy = self.__test_metric_network(trained_net, dataset)
        return final_accuracy


class InceptionScore:
    """
    Concrete Class for computing the Inception Score over a set of given probabilities. As proposed in:
    https://proceedings.neurips.cc/paper_files/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf.
    As suggested from the authors, this metric should be evaluated only on a large enough number of samples
    (i.e., >= 50k).

    Attributes
    ----------
    num_splits : int, Optional
        Number of splits to use for the probabilities.

    """

    def __init__(self, num_splits: int = 10):
        self.num_splits = num_splits

    def calculate(self, probabilities_df: pandas.DataFrame) -> (float, float):
        """
        Compute the inception score over a set of probabilities provided as parameter.

        Parameters
        ----------
        probabilities_df : pandas.DataFrame
            Set of probabilities over which to compute the inception score.

        Returns
        -------
        out : typing.Tuple
            Contains the mean and the standard deviation of the computed scores.

        """

        probabilities = probabilities_df.to_numpy()
        num_probabilities = probabilities.shape[0]
        scores = []
        for i in range(self.num_splits):
            part = probabilities[
                   i * num_probabilities // self.num_splits: (i + 1) * num_probabilities // self.num_splits]
            kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))

        return float(np.mean(scores)), float(np.std(scores))
