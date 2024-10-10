# SynthVal

**SynthVal** is a Python package developed to validate and verify the quality of synthetically generated data by comparing it to original data. The project focuses primarily on medical images, such as chest x-rays and mammographies, offering tools to compute similarity measures between original and synthetic datasets.

### Purpose

With the growing use of synthetic data in fields like healthcare and AI, it is essential to have reliable methods to evaluate how closely synthetic data resembles real data. SynthVal addresses this need by providing a straightforward framework for comparing original and synthetic data, enabling users to assess the quality and fidelity of synthetic datasets.

### Key Features

SynthVal is built around two main modules:

1. **Feature Extraction**: The `features_extraction.py` module extracts vectors of features from images, capturing their essential characteristics to serve as the basis for similarity comparison.
   
2. **Similarity Metrics**: The `metrics.py` module calculates several metrics to determine the similarity between original and synthetic datasets. These include:
   - **Energy Distance**: Measures the distance between the distributions of the datasets.
   - **Kullback-Leibler Divergence**: Quantifies the divergence between two probability distributions.
   - **ML-Based Accuracy**: Uses machine learning models to assess how well the synthetic data imitates the original by training models to distinguish between the two.

### Approach

SynthVal approaches the comparison between original and synthetic data by focusing on the underlying probability distributions rather than raw data points. This probabilistic method provides more insightful and interpretable similarity measures, especially for medical images where capturing critical features is vital.