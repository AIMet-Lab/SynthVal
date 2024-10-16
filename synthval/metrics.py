"""
Module for computing various similarity metrics between two sets of samples originating from multivariate distributions.

This module defines abstract and concrete classes for computing similarity metrics between samples from two
distributions. The available metrics include Kullback-Leibler divergence, Wasserstein distance, Energy distance,
Mean Mahalanobis distance, Frechet Distance, Inception Score, Kernel Distances and others.
It should be noted that dist_p usually represent the reference distribution, whereas dist_q represent the distribution
we wish to evaluate.

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
InceptionScore()
    Evaluation metrics for generated images.
FrechetDistance(SimilarityMetric)
    Concrete implementation of the Frechet distance.
KernelDistance(SimilarityMetric)
    Concrete implementation of the Kernel distance.
PRScore(SimilarityMetric)
    Concrete implementation of the Precision and Recall Scores.

"""

import abc

import numpy
import numpy.linalg
import scipy.spatial as sci_sp
import scipy.stats as sci_stats
import scipy
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
    Pérez-Cruz, F. - Kullback-Leibler divergence estimation of continuous distributions - IEEE International Symposium
    on Information Theory, 2008.

    """

    def __init__(self):
        SimilarityMetric.__init__(self)

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
        return numpy.log(s / r).sum() * d / n + numpy.log(m / (n - 1.0))


class WassersteinDistance(SimilarityMetric):
    """
    Similarity Metric computing the Wasserstein Distance.

    """

    def __init__(self):
        SimilarityMetric.__init__(self)

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
        SimilarityMetric.__init__(self)

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
        SimilarityMetric.__init__(self)

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

        covariance_matrix = numpy.cov(dist_q, rowvar=False)
        mean_vector = numpy.mean(dist_q, axis=0)

        m_distances = []
        for i in range(dist_p.shape[0]):
            m_distances.append(sci_sp.distance.mahalanobis(dist_p[i, :], mean_vector, covariance_matrix))

        m_distances = numpy.array(m_distances)
        return numpy.mean(m_distances)


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

        SimilarityMetric.__init__(self)

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
        c_weights = sklearn.utils.class_weight.compute_class_weight("balanced", classes=numpy.unique(labels), y=labels)
        c_weights = numpy.float32(c_weights)
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
    Class for computing the Inception Score over a set of probabilities.

    The Inception Score measures the quality of images generated by generative models (such as GANs) by evaluating the
    entropy of predictions made by the Inception model. Higher scores correspond to generated images that are both
    diverse and sharp. As suggested by the authors of the reference paper, this metric should be computed over a
    sufficiently large number of samples (at least 50,000).

    Attributes
    ----------
    num_splits : int, Optional
        Number of splits to use for the probabilities. This is used to compute the score on different subsets of
        the data and obtain a more robust estimate of the score (default: 10).

    References
    ----------
    Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen - Improved Techniques for
    Training GANs - Annual Conference on Neural Information Processing Systems, 2016.

    """

    def __init__(self, num_splits: int = 10):
        self.num_splits = num_splits

    def calculate(self, probabilities_df: pandas.DataFrame) -> (float, float):
        """
        Compute the Inception Score over the provided probabilities.

        The score is computed by evaluating the Kullback-Leibler (KL) divergence between the conditional class
        distribution for each generated image and the marginal class distribution across the dataset. The exponential
        of the average KL divergence is the Inception Score. The mean and standard deviation of the score are calculated
        over multiple splits of the data for robustness.

        Parameters
        ----------
        probabilities_df : pandas.DataFrame
            DataFrame containing the predicted class probabilities for each generated sample. The probabilities are
            typically obtained from a pre-trained Inception model.

        Returns
        -------
        score_mean : float
            The mean of the computed Inception Scores across the splits.
        score_std : float
            The standard deviation of the Inception Scores across the splits.
        """

        # Convert the DataFrame to a NumPy array for processing
        probabilities = probabilities_df.to_numpy()
        num_probabilities = probabilities.shape[0]

        # List to store scores computed over each split
        scores = []

        # Calculate the score for each subset of data (split for robustness)
        for i in range(self.num_splits):
            # Select a subset of the data based on the current split
            subset = probabilities[
                i * num_probabilities // self.num_splits : (i + 1) * num_probabilities // self.num_splits
            ]

            # Compute the KL divergence for the current subset
            # KL(p(y|x) || p(y)) where p(y|x) is the per-sample probability and p(y) is the marginal probability
            kl_divergence = subset * (numpy.log(subset) - numpy.log(numpy.mean(subset, axis=0, keepdims=True)))

            # Average the KL divergence for the subset and exponentiate the result to get the Inception Score
            kl_mean = numpy.mean(numpy.sum(kl_divergence, axis=1))
            scores.append(numpy.exp(kl_mean))

        # Compute the mean and standard deviation of the Inception Scores across all splits
        score_mean = float(numpy.mean(scores))
        score_std = float(numpy.std(scores))

        return score_mean, score_std


class FrechetDistance(SimilarityMetric):
    """
    Similarity Metric that computes the Frechet distance (also known as Fréchet Inception Distance) between two
    distributions by comparing their means and covariances.

    When applied to features extracted from the last average pooling layer of the Inception model (e.g., those provided
    by synthval.features_extraction.InceptionExtractor), it corresponds to the standard Frechet Inception Distance.


    Attributes
    ----------
    eps : float, Optional
        Small offset added to the covariance matrices to handle numerical issues such as matrix singularity
        (default: 1e-6).


    References
    ----------
    Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter - GANs Trained by a Two
    Time-Scale Update Rule Converge to a Local Nash Equilibrium - Annual Conference on Neural Information
    Processing Systems, 2016.

    """

    def __init__(self, eps: float = 1e-6):
        SimilarityMetric.__init__(self)
        self.eps = eps

    def calculate(self, dist_p_df: pandas.DataFrame, dist_q_df: pandas.DataFrame) -> float:
        """
        Compute the Frechet distance between two distributions by comparing the mean and covariance of the samples
        provided.

        Parameters
        ----------
        dist_p_df : pandas.DataFrame
            Set of samples representing distribution P.
        dist_q_df : pandas.DataFrame
            Set of samples representing distribution Q.

        Returns
        -------
        out : float
        The Frechet Distance.

        """

        # Convert dataframes to NumPy arrays for computation
        dist_p = dist_p_df.to_numpy()
        dist_q = dist_q_df.to_numpy()

        # Compute the mean of the samples for both distributions
        mean_dist_p = numpy.mean(dist_p, axis=0)
        mean_dist_q = numpy.mean(dist_q, axis=0)

        # Compute the covariance matrices for both distributions
        cov_dist_p = numpy.cov(dist_p, rowvar=False)
        cov_dist_q = numpy.cov(dist_q, rowvar=False)

        # Compute the difference between the means of the two distributions
        mean_diff = mean_dist_q - mean_dist_p

        # Compute the square root of the product of the covariance matrices
        # Note: The function returns both the matrix and a flag; we only need the matrix here
        cov_mean, _ = scipy.linalg.sqrtm(numpy.dot(cov_dist_q, cov_dist_p), disp=False)

        # Handle potential numerical issues (e.g., if cov_mean has non-finite values)
        if not numpy.isfinite(cov_mean).all():
            # If cov_mean is not finite, add a small offset (eps) to the diagonal of the covariance matrices
            offset = numpy.eye(cov_dist_q.shape[0]) * self.eps
            cov_mean = scipy.linalg.sqrtm(numpy.dot(cov_dist_q + offset, cov_dist_p + offset))

        # If there are small imaginary components due to numerical errors, discard the imaginary part
        if numpy.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real

        # Compute the trace of the square root of the product of the covariance matrices
        tr_cov_mean = numpy.trace(cov_mean)

        # Calculate the Frechet distance using the formula:
        # ||mean_diff||^2 + Tr(cov_p) + Tr(cov_q) - 2 * Tr(sqrt(cov_p * cov_q))
        f_dist = (numpy.dot(mean_diff, mean_diff)  # Squared difference of the means
                  + numpy.trace(cov_dist_q)  # Trace of the covariance of Q
                  + numpy.trace(cov_dist_p)  # Trace of the covariance of P
                  - 2 * tr_cov_mean)  # 2 times the trace of the product of the covariances

        return f_dist


class KernelDistance(SimilarityMetric):
    """
    Similarity Metric that computes the Kernel distance between two distributions using provided samples.

    If the features extracted are from the last average pooling layer of the Inception model (e.g., from
    synthval.features_extraction.InceptionExtractor), this metric corresponds to the standard Kernel Inception
    Distance (KID).

    Attributes
    ----------
    max_samples: int, Optional
        Max number of samples to consider in the computation. To use in the case of limited time or hardware
        capabilities (default: 1000000).
    num_subsets: int, Optional
        Number of subset to use for calculating the Kernel distance (default: 100).
    max_subset_size: int, Optional
        Maximum size of each subset (default: 1000).

    References
    ----------
    Mikolaj Binkowski, Danica J. Sutherland, Michael Arbel, Arthur Gretton - Demystifying MMD GANs - 6th International
    Conference on Learning Representations, 2018.

    """

    def __init__(self, max_samples: int = 1000000, num_subsets: int = 100, max_subset_size: int = 1000):
        SimilarityMetric.__init__(self)
        self.max_samples = max_samples
        self.num_subsets = num_subsets
        self.max_subset_size = max_subset_size

    def calculate(self, dist_p_df: pandas.DataFrame, dist_q_df: pandas.DataFrame) -> float:
        """
        Compute the Kernel distance between two distributions using the samples
        provided.

        Parameters
        ----------
        dist_p_df : pandas.DataFrame
            Set of samples representing distribution P.
        dist_q_df : pandas.DataFrame
            Set of samples representing distribution Q.

        Returns
        -------
        out : float
        The Kernel Distance.

        """

        # Convert the dataframes to NumPy arrays for computation
        dist_p = dist_p_df.to_numpy()
        dist_q = dist_q_df.to_numpy()

        # Limit the number of samples for both distributions to max_samples if needed
        if dist_p.shape[0] > self.max_samples:
            selected_indices = numpy.random.choice(dist_p.shape[0], self.max_samples, replace=False)
            dist_p = dist_p[selected_indices]

        if dist_q.shape[0] > self.max_samples:
            selected_indices = numpy.random.choice(dist_q.shape[0], self.max_samples, replace=False)
            dist_q = dist_q[selected_indices]

        # Number of features (dimensions) in the samples
        num_features = dist_p.shape[1]

        # Determine the subset size for kernel computation (limited by max_subset_size)
        subset_size = min(min(dist_p.shape[0], dist_q.shape[0]), self.max_subset_size)

        # Initialize accumulator for the kernel distance computation
        total_kernel_distance = 0

        # Perform the Kernel distance computation over multiple subsets
        for _ in range(self.num_subsets):
            # Randomly sample subsets from both distributions
            subset_p = dist_p[numpy.random.choice(dist_p.shape[0], subset_size, replace=False)]
            subset_q = dist_q[numpy.random.choice(dist_q.shape[0], subset_size, replace=False)]

            # Compute pairwise distances between samples within subsets
            # Matrix operations for subsets of P and Q
            kernel_aa = (subset_p @ subset_p.T / num_features + 1) ** 3  # Kernel for P
            kernel_bb = (subset_q @ subset_q.T / num_features + 1) ** 3  # Kernel for Q
            kernel_ab = (subset_p @ subset_q.T / num_features + 1) ** 3  # Kernel between P and Q

            # Sum over the elements of kernel matrices, excluding diagonal elements
            total_kernel_distance += (
                    (kernel_aa.sum() - numpy.diag(kernel_aa).sum()) / (subset_size - 1)  # Within P
                    + (kernel_bb.sum() - numpy.diag(kernel_bb).sum()) / (subset_size - 1)  # Within Q
                    - 2 * kernel_ab.sum() / subset_size  # Between P and Q
            )

        # Compute the final kernel distance by averaging across all subsets
        kernel_distance = total_kernel_distance / self.num_subsets / subset_size
        return kernel_distance


class PRScores(SimilarityMetric):
    """
    A Similarity Metric class that computes the Precision and Recall scores between two distributions
    (dist_p and dist_q).

    Attributes
    ----------
    row_batch_size: int, optional
        Size of the row batches used when computing pairwise distances. This provides a trade-off between memory
        usage and performance (default: 25000).
    col_batch_size: int, optional
        Size of the column batches used for computing pairwise distances (default: 50000).
    num_nearest_n: int, optional
        Number of nearest neighbors used to estimate the manifold. The manifold is used for computing precision
        and recall (default: 3).

    References
    ----------
    Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, Timo Aila - Improved Precision and Recall
    Metric for Assessing Generative Models - Annual Conference on Neural Information Processing Systems, 2019.

    """

    def __init__(self, row_batch_size: int = 25000, col_batch_size: int = 50000, num_nearest_n: int = 3):
        SimilarityMetric.__init__(self)
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.num_nearest_n = num_nearest_n

        # Detect if CUDA or MPS (Apple Silicon) is available and default to CPU if not
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def calculate(self, dist_p_df: pandas.DataFrame, dist_q_df: pandas.DataFrame) -> (float, float):
        """
        Compute Precision and Recall metrics between two distributions.

        Parameters
        ----------
        dist_p_df : pandas.DataFrame
            DataFrame containing samples from distribution P.
        dist_q_df : pandas.DataFrame
            DataFrame containing samples from distribution Q.

        Returns
        -------
        precision : float
            Precision metric between distribution P and Q.
        recall : float
            Recall metric between distribution P and Q.
        """

        # Convert DataFrames to NumPy arrays for distance computation
        dist_p = dist_p_df.to_numpy()
        dist_q = dist_q_df.to_numpy()

        results = {}

        # Compute precision and recall in two passes:
        for name, manifold, probes in [('precision', dist_p, dist_q),
                                       ('recall', dist_q, dist_p)]:
            kth = []  # To store the k-th nearest distances for manifold points

            # Process manifold in batches to control memory usage
            for manifold_batch in numpy.split(manifold, self.row_batch_size):
                # Compute pairwise distances between batches, using available GPU/CPU/MPS
                dist = self.__compute_distances(
                    row_features=torch.tensor(manifold_batch, dtype=torch.float32).to(self.device),
                    col_features=torch.tensor(manifold, dtype=torch.float32).to(self.device),
                    col_batch_size=self.col_batch_size
                )

                # Check if we are on MPS and move to CPU to compute kthvalue
                if self.device.type == "mps":
                    kth_cpu = dist.to(torch.float32).cpu().kthvalue(self.num_nearest_n + 1).values.to(torch.float16)
                    kth.append(kth_cpu.to(self.device))
                else:
                    kth.append(dist.to(torch.float32).kthvalue(self.num_nearest_n + 1).values.to(torch.float16))

            # Concatenate k-th nearest distances for all batches
            kth = torch.cat(kth)

            # To store the boolean results of whether points in probes are within the manifold's
            # k-th nearest distance
            pred = []

            # Process probe data in batches as well
            for probes_batch in numpy.split(probes, self.row_batch_size):
                dist = self.__compute_distances(
                    row_features=torch.tensor(probes_batch, dtype=torch.float32).to(self.device),
                    col_features=torch.tensor(manifold, dtype=torch.float32).to(self.device),
                    col_batch_size=self.col_batch_size
                )
                # Check if any distance is within the k-th nearest neighbors
                pred.append((dist <= kth).any(dim=1))

            # Calculate the mean of the boolean results (True means within k-nearest distance, thus a "hit")
            results[name] = float(torch.cat(pred).to(torch.float32).mean())

        # Return precision and recall
        return results['precision'], results['recall']

    @staticmethod
    def __compute_distances(row_features, col_features, col_batch_size):
        """
        Compute pairwise distances between row and column features using batches to optimize memory usage.

        Parameters
        ----------
        row_features : torch.Tensor
            Tensor of feature vectors representing the rows.
        col_features : torch.Tensor
            Tensor of feature vectors representing the columns.
        col_batch_size : int
            Size of the column batches for computing distances.

        Returns
        -------
        dist_batches : torch.Tensor
            Tensor of pairwise distances.
        """

        # Get the number of columns (features in the distribution)
        num_cols = col_features.shape[0]

        # Calculate the number of column batches needed
        num_batches = (num_cols - 1) // col_batch_size + 1

        # Split columns into batches
        col_batches = torch.split(col_features, col_batch_size)

        dist_batches = []

        # Loop through column batches processed by the GPU/CPU/MPS
        for col_batch in col_batches:
            # Compute pairwise distances between row features and column batch
            dist_batch = torch.cdist(row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
            dist_batches.append(dist_batch)

        # Concatenate the distance batches along columns
        return torch.cat(dist_batches, dim=1)
