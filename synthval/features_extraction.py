"""
Module for feature extraction using various models from HuggingFace, including Rad-Dino, DinoV2, and MambaVision.

This module defines abstract and concrete classes for feature extraction from images, leveraging different pre-trained
models available through the HuggingFace library. The module supports extracting features from images using models such
as Rad-Dino, DinoV2, and MambaVision, with each extractor providing methods for single and batch feature extraction.

Classes
-------
FeatureExtractor(abc.ABC)
    Abstract base class for defining a feature extractor interface.
RadDinoFeatureExtractor(FeatureExtractor)
    Concrete feature extractor using the HuggingFace Rad-Dino model.
DinoV2FeatureExtractor(FeatureExtractor)
    Concrete feature extractor using models from the HuggingFace DinoV2 family.
MambaFeatureExtractor(FeatureExtractor)
    Concrete feature extractor using models from the HuggingFace MambaVision family.
InceptionExtractor(FeatureExtractor)
    Concrete feature extractor using traditional Inception models from the timm library.

"""

import abc
import os
from typing import List
import PIL.Image
import pandas
import numpy
import torch
import transformers
import timm
import timm.data
import timm.data.transforms_factory
import synthval.utilities


class FeatureExtractor(abc.ABC):
    """
    Abstract base class representing a generic feature extractor.
    Child classes must implement the batch_feature_extraction method for batch processing.
    """

    @abc.abstractmethod
    def batch_feature_extraction(self, images: List[PIL.Image.Image]) -> numpy.ndarray:
        """
        Abstract method to extract features from a batch of images.

        Parameters
        ----------
        images : List[PIL.Image.Image]
            List of PIL images to extract features from.

        Returns
        -------
        numpy.ndarray
            A NumPy array containing extracted features for the batch.
        """
        raise NotImplementedError

    def group_feature_extraction(self, source_folder_path: str, batch_size: int = 16,
                                 verbose: bool = True) -> pandas.DataFrame:
        """
        Extract features from all images in a specified folder using batch processing.

        Parameters
        ----------
        source_folder_path : str
            Path to the folder containing images.
        batch_size : int, optional
            Number of images to process in a single batch (default: 16).
        verbose : bool, optional
            If True, logs progress during extraction (default: True).

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing extracted features for all images.
        """
        stream_logger = synthval.utilities.get_stream_logger("synthval.feature_extraction") if verbose else None
        image_paths = sorted([os.path.join(source_folder_path, img) for img in os.listdir(source_folder_path)])
        features_dataset = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            if stream_logger:
                stream_logger.info(f"Processing batch {i//batch_size + 1}/{-(-len(image_paths) // batch_size)}")
            batch_features = self.batch_feature_extraction([synthval.utilities.get_pil_image(p) for p in batch_paths])
            features_dataset.append(batch_features)

        # Stack all batch results into a single numpy array
        features_dataset = numpy.vstack(features_dataset)
        return pandas.DataFrame(features_dataset)

    def get_features_df(self, source_folder_path: str, save_path: str = None, batch_size: int = 16,
                        verbose: bool = True) -> pandas.DataFrame:
        """
        Extract features from images in a folder and optionally save them to a CSV file.

        Parameters
        ----------
        source_folder_path : str
            Path to the folder containing images.
        save_path : str, optional
            Path to save the features DataFrame as a CSV file. If a CSV file exists, it will be loaded instead.
        batch_size : int, optional
            Number of images to process per batch (default: 16).
        verbose : bool, optional
            If True, logs progress during extraction (default: True).

        Returns
        -------
        pandas.DataFrame
            DataFrame containing extracted features for all images.
        """
        if save_path and os.path.exists(save_path):
            return pandas.read_csv(save_path, header=None)

        features_df = self.group_feature_extraction(source_folder_path, batch_size, verbose)
        if save_path:
            features_df.to_csv(save_path, index=False, header=False)
        return features_df


class RadDinoFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using the HuggingFace model microsoft/rad-dino for extracting features from images.
    This implementation supports batch processing for improved performance on large datasets.
    """

    def __init__(self):
        """
        Initializes the Rad-Dino feature extractor by loading the pre-trained model and processor.
        """
        super().__init__()
        self.repo = "microsoft/rad-dino"
        self.processor = transformers.AutoImageProcessor.from_pretrained(self.repo)
        self.model = transformers.AutoModel.from_pretrained(self.repo)

        # Selecting the device for computation (CUDA, MPS, or CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        self.model.to(self.device)

    def batch_feature_extraction(self, images: List[PIL.Image.Image]) -> numpy.ndarray:
        """
        Extract features from a batch of images using the HuggingFace Rad-Dino model.

        Parameters
        ----------
        images : List[PIL.Image.Image]
            List of PIL images to extract features from.

        Returns
        -------
        numpy.ndarray
            A NumPy array where each row corresponds to the feature vector of an image.
        """

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        # Perform model inference with no gradient tracking (for efficiency)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Extract CLS token embeddings as feature representations
        cls_embeddings = outputs.pooler_output

        # Convert to NumPy format and return
        return cls_embeddings.detach().cpu().numpy()


class DinoV2FeatureExtractor(FeatureExtractor):
    """
    Feature extractor using models from the HuggingFace DinoV2 family
    (https://huggingface.co/collections/facebook/dinov2-6526c98554b3d2576e071ce3).

    Note: As of December 16, 2024, Dino models utilize the operator `aten::upsample_bicubic2d.out`,
    which is not currently supported on the MPS backend. To use these models as feature extractors
    on MPS, users must set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK` to `1`.
    """

    def __init__(self, model_id: str):
        """
        Initializes the DinoV2 feature extractor by loading the pre-trained model and processor.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID for the selected DinoV2 model.
        """
        super().__init__()
        self.model_id = model_id
        self.processor = transformers.AutoImageProcessor.from_pretrained(self.model_id)
        self.model = transformers.AutoModel.from_pretrained(self.model_id)

        # Selecting the device for computation (CUDA, MPS, or CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        )
        self.model.to(self.device)

    def batch_feature_extraction(self, images: List[PIL.Image.Image]) -> numpy.ndarray:
        """
        Extract features from a batch of images using the HuggingFace DinoV2 model.

        Parameters
        ----------
        images : List[PIL.Image.Image]
            List of PIL images to extract features from.

        Returns
        -------
        numpy.ndarray
            A NumPy array where each row corresponds to the feature vector of an image.
        """
        # Preprocess images using the processor
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        # Perform model inference with no gradient tracking (for efficiency)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Extract CLS token embeddings as feature representations
        cls_embeddings = outputs.pooler_output

        # Convert to NumPy format and return
        return cls_embeddings.detach().cpu().numpy()


class MambaFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using models from the HuggingFace MambaVision family
    (https://huggingface.co/collections/nvidia/mambavision-66943871a6b36c9e78b327d3).

    Note: MambaVision models require a CUDA-enabled environment and the installation of specific packages.
    To use these models as feature extractors, users must have CUDA installed and install the necessary
    packages by running `pip install mambavision`.
    """

    def __init__(self, model_id: str):
        """
        Initializes the MambaVision feature extractor by loading the pre-trained model.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID for the selected MambaVision model.
        """
        super().__init__()
        self.model_id = model_id
        self.model = transformers.AutoModel.from_pretrained(self.model_id, trust_remote_code=True)

        # Selecting the device for computation (CUDA required for MambaVision models)
        if not torch.cuda.is_available():
            raise RuntimeError("MambaVision models require a CUDA-enabled environment.")

        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.model.eval()

    def batch_feature_extraction(self, images: List[PIL.Image.Image]) -> numpy.ndarray:
        """
        Extract features from a batch of images using the HuggingFace MambaVision model.

        Parameters
        ----------
        images : List[PIL.Image.Image]
            List of PIL images to extract features from.

        Returns
        -------
        numpy.ndarray
            A NumPy array where each row corresponds to the feature vector of an image.
        """
        # Convert images to RGB as MambaVision models require 3 channels
        images = [img.convert("RGB") for img in images]

        # Define input resolution based on the first image's size
        input_resolution = (3, images[0].width, images[0].height)

        # Prepare image transformations based on the model configuration
        transform = timm.data.transforms_factory.create_transform(
            input_size=input_resolution,
            is_training=False,
            mean=self.model.config.mean,
            std=self.model.config.std,
            crop_mode=self.model.config.crop_mode,
            crop_pct=self.model.config.crop_pct
        )

        # Apply transformation to all images and convert to tensor
        inputs = torch.stack([transform(img).unsqueeze(0) for img in images]).to(self.device)

        # Perform model inference with no gradient tracking (for efficiency)
        with torch.no_grad():
            out_avg_pool, _ = self.model(inputs)

        # Convert output to NumPy format and return
        return out_avg_pool.cpu().numpy()


class InceptionExtractor(FeatureExtractor):
    """
    Feature extractor using traditional Inception models from the timm library (https://huggingface.co/docs/timm/index).
    Supports batch processing and provides options to extract feature embeddings or class probabilities.
    """

    def __init__(self, model_id: str, get_probabilities: bool = False):
        """
        Initializes the Inception feature extractor by loading the pre-trained model.

        Parameters
        ----------
        model_id : str
            Timm model ID for the selected Inception model (e.g., 'inception_v3', 'inception_v4').
        get_probabilities : bool, optional
            If True, extracts model predictions instead of feature embeddings (default: False).
        """
        super().__init__()
        self.model_id = model_id
        self.get_probabilities = get_probabilities

        # Load the Inception model from timm with appropriate output configuration
        self.model = timm.create_model(
            self.model_id, pretrained=True, num_classes=1000 if get_probabilities else 0
        )
        self.model.eval()

        # Selecting the device for computation
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Prepare transformation configuration
        self.config = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.transforms_factory.create_transform(**self.config)

    def batch_feature_extraction(self, images: List[PIL.Image.Image]) -> numpy.ndarray:
        """
        Extract features (or probabilities) from a batch of images using the Inception model.

        Parameters
        ----------
        images : List[PIL.Image.Image]
            List of PIL images to extract features from.

        Returns
        -------
        numpy.ndarray
            A NumPy array where each row corresponds to the feature vector (or probability vector) of an image.
        """
        # Convert images to RGB as Inception models expect 3 channels
        images = [img.convert("RGB") for img in images]

        # Apply transformations and stack images into a single tensor
        inputs = torch.stack([self.transform(img) for img in images]).to(self.device)

        # Perform model inference with no gradient tracking
        with torch.no_grad():
            outputs = self.model(inputs)

        # Apply softmax if extracting probabilities
        if self.get_probabilities:
            outputs = torch.nn.functional.softmax(outputs, dim=1)

        # Convert to NumPy format and return
        return outputs.cpu().numpy()

