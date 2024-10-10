"""
Module providing utility functions and classes for handling image datasets and feature extraction.

This module contains utility functions for image processing, such as converting DICOM and other image formats to
PIL Images, and managing datasets of extracted features for training and testing. The dataset management is
integrated with pyNeVer's training and testing strategies.

Functions
---------
get_stream_logger(logger_origin: str) -> logging.Logger
    Utility function to instantiate a stream logger.
get_pil_image(image_path: str) -> PIL.Image.Image
    Utility function to convert a DICOM or generic image file to a PIL Image.

Classes
-------
FeaturesDataset(pynever.datasets.Dataset)
    A dataset class for managing features extracted from original and synthetically generated images for use in
    PyNEVER training and testing.

"""


import sys

import PIL.Image
import pandas
import math
import pynever.datasets
import logging
import pydicom
import numpy


def get_stream_logger(logger_origin: str) -> logging.Logger:
    """
    Utility function to instantiate a stream logger.

    Parameters
    ----------
    logger_origin: str
        Origin of the logger.

    Returns
    -------
    logging.Logger
        The stream logger.
    """

    logger = logging.getLogger(logger_origin)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


def get_pil_image(image_path: str) -> PIL.Image.Image:

    """
    Utility function to convert a .dcm or generic image to a PIL Image.

    Parameters
    ----------
    image_path: str
        Path to the image to convert.

    Returns
    -------
    PIL.Image
        The converted PIL Image.
    """

    if image_path.split(".")[-1] == "dcm":

        dcm_image = pydicom.dcmread(image_path)
        new_image = dcm_image.pixel_array.astype(float)  # Convert the values into float

        scaled_image = (numpy.maximum(new_image, 0) / new_image.max()) * 255.0
        scaled_image = numpy.uint8(scaled_image)

        final_image = PIL.Image.fromarray(scaled_image)
    else:
        final_image = PIL.Image.open(image_path)

    return final_image


class FeaturesDataset(pynever.datasets.Dataset):

    """
    Utility class for the management of the dataset composed by the feature extracted from the original images
    and the synthetically generated images. Needed for using the pynever training and testing strategies.

    Attributes
    ----------
    train_df : pandas.Dataframe
        The training set.
    test_df : pandas.Dataframe
        The test set.
    train_mode : bool
        Flag controlling if the __get_item__ function returns element from the training set or the test set.

    """

    def __init__(self, ori_features: pandas.DataFrame, gen_features: pandas.DataFrame, test_percentage: float = 0.2,
                 rng_seed: int = 0, train_mode: bool = True):

        """
        Initialization methods for the Dataset.

        Parameters
        ----------
        ori_features : pandas.Dataframe
            Should contain the features extracted from the original images.
        gen_features : pandas.Dataframe
            Should contain the features extracted from the synthetically generated images.
        test_percentage : float, Optional
            Percentage of the dataset that should be reserved for testing - that is, computing the final similarity
            metric (default: 0.2).
        rng_seed : int, Optional
            Random seed for numpy.random utilities (default: 0)
        train_mode : bool, Optional
            Flag controlling if the __get_item__ function returns element from the training set or the test set.
        """

        rng = numpy.random.default_rng(seed=rng_seed)

        # Add label to features dfs
        ori_features.insert(len(ori_features.columns), "Label",
                            numpy.array([0 for i in range(ori_features.__len__())], int))
        gen_features.insert(len(gen_features.columns), "Label",
                            numpy.array([1 for i in range(gen_features.__len__())], int))

        ori_test_size = math.floor(ori_features.__len__() * test_percentage)
        gen_test_size = math.floor(gen_features.__len__() * test_percentage)

        ori_test_indexes = rng.choice(ori_features.__len__(), ori_test_size, replace=False)
        ori_train_indexes = [index for index in range(ori_features.__len__()) if index not in ori_test_indexes]

        gen_test_indexes = rng.choice(gen_features.__len__(), gen_test_size, replace=False)
        gen_train_indexes = [index for index in range(gen_features.__len__()) if index not in gen_test_indexes]

        ori_train_df = ori_features.iloc[ori_train_indexes]
        gen_train_df = gen_features.iloc[gen_train_indexes]
        ori_test_df = ori_features.iloc[ori_test_indexes]
        gen_test_df = gen_features.iloc[gen_test_indexes]

        self.train_df = pandas.concat([ori_train_df, gen_train_df], axis="index", ignore_index=True)
        self.test_df = pandas.concat([ori_test_df, gen_test_df], axis="index", ignore_index=True)

        self.train_df = self.train_df.sample(frac=1).reset_index(drop=True)
        self.test_df = self.test_df.sample(frac=1).reset_index(drop=True)

        self.train_mode = train_mode

    def __getitem__(self, index: int):
        if self.train_mode:
            return (self.train_df.iloc[index][self.train_df.columns[:-1]].to_numpy(numpy.float32),
                    int(self.train_df.iloc[index][self.train_df.columns[-1]]))
        else:
            return (self.test_df.iloc[index][self.test_df.columns[:-1]].to_numpy(numpy.float32),
                    int(self.test_df.iloc[index][self.test_df.columns[-1]]))

    def __len__(self):
        if self.train_mode:
            return self.train_df.__len__()
        else:
            return self.test_df.__len__()