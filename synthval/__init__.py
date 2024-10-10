"""
Package designed to validate the quality of synthetically generated data, with a focus on medical images like
chest x-rays and mammographies, by providing tools for feature extraction and similarity metric calculations to
compare original and synthetic datasets.

The config.py submodule provides default configurations parameters for setting up neural networks and training them
using the pyNeVer (https://github.com/NeVerTools/pyNeVer) tool.

The features_extraction.py submodule defines abstract and concrete classes for feature extraction from images.

The metrics.py submodule defines abstract and concrete classes for computing similarity metrics between samples
from two distributions.

Finally, the utilities.py submodule contains utility functions for image processing, such as converting
DICOM and other image formats to PIL Images, and managing datasets of extracted features for training and testing.

"""

__all__ = ["configs", "features_extraction", "metrics", "utilities"]