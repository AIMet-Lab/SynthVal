import sys

import PIL.Image
import logging
import pydicom
import numpy as np


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

        dcm_image = pydicom.read_file(image_path)
        new_image = dcm_image.pixel_array.astype(float)  # Convert the values into float

        scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
        scaled_image = np.uint8(scaled_image)

        final_image = PIL.Image.fromarray(scaled_image)
    else:
        final_image = PIL.Image.open(image_path)

    return final_image
