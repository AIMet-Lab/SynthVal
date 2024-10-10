import abc
import os

import PIL.Image
import pandas
import synthval.utilities as utilities

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
import timm.data.transforms_factory


class FeatureExtractor(abc.ABC):
    """
    Abstract base class representing a generic feature extractor. Child classes must implement
    the concrete `feature_extraction` method for specific feature extraction.

    Methods
    -------
    feature_extraction(image: PIL.Image.Image) -> np.ndarray
        Extract relevant features from a PIL image, returning them as a NumPy array.
    group_feature_extraction(source_folder_path: str, verbose: bool = True) -> pandas.DataFrame
        Extract features from a dataset of images in a specified folder.
    get_features_df(source_folder_path: str, save_path: str = None, verbose: bool = True) -> pandas.DataFrame
        Extract features from images in a folder, with an option to save or load a CSV file.
    """

    @abc.abstractmethod
    def feature_extraction(self, image: PIL.Image.Image) -> np.ndarray:
        """
        Abstract method to extract features from a PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
            The image from which features are to be extracted.

        Returns
        -------
        np.ndarray
            A NumPy array containing the extracted features.
        """
        raise NotImplementedError

    def group_feature_extraction(self, source_folder_path: str, verbose: bool = True) -> pandas.DataFrame:
        """
        Extract features from all images in the specified folder.

        Parameters
        ----------
        source_folder_path : str
            The path to the folder containing the images.
        verbose : bool, optional
            If True, log the progress of feature extraction (default: True).

        Returns
        -------
        pandas.DataFrame
            A DataFrame where each row represents the features of an image.
        """

        # Set up logger if verbosity is enabled
        stream_logger = None
        if verbose:
            stream_logger = utilities.get_stream_logger("synthval.feature_extraction")

        # Retrieve image IDs from the folder
        images_ids = sorted(os.listdir(source_folder_path))
        features_dataset = []

        # Iterate through images and extract features
        for image_id in images_ids:
            if stream_logger is not None:
                stream_logger.info(f"Extracting Features from Image: {image_id}.")

            # Construct the full path to the image
            image_path = os.path.join(source_folder_path, image_id)

            # Load the image as a PIL object
            pil_image = utilities.get_pil_image(image_path)

            # Extract features from the image
            np_features = self.feature_extraction(pil_image)

            # Append the features to the dataset
            features_dataset.append(np_features)

        # Convert the list of features to a NumPy array and then to a DataFrame
        features_dataset = np.array(features_dataset).squeeze()
        features_df = pandas.DataFrame(features_dataset)

        return features_df

    def get_features_df(self, source_folder_path: str, save_path: str = None, verbose: bool = True) -> pandas.DataFrame:
        """
        Extract features from a dataset of images and optionally save them to a CSV file.

        Parameters
        ----------
        source_folder_path : str
            Path to the folder containing the images.
        save_path : str, optional
            Path to save the features DataFrame as a CSV file. If a CSV file already exists at
            the provided path, it will be loaded instead of recalculating features (default: None).
        verbose : bool, optional
            If True, log the progress of feature extraction (default: True).

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the extracted features.
        """

        # Check if a saved CSV already exists
        if save_path is not None and os.path.exists(save_path):
            features_df = pandas.read_csv(save_path, header=None)
        else:
            # Extract features from images if CSV does not exist
            features_df = self.group_feature_extraction(source_folder_path=source_folder_path, verbose=verbose)

            # Save the DataFrame to a CSV file if a save path is provided
            if save_path is not None:
                features_df.to_csv(save_path, sep=",", index=False, header=False)

        return features_df


class RadDinoFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using the HuggingFace model microsoft/rad-dino for extracting features from images.

    Methods
    -------
    feature_extraction(image: PIL.Image.Image) -> np.ndarray
        Extract features from an image using the Rad-Dino model.
    """

    def __init__(self):
        super().__init__()

    def feature_extraction(self, image: PIL.Image.Image) -> np.ndarray:
        """
        Extract features from a PIL image using the HuggingFace Rad-Dino model.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to extract features from.

        Returns
        -------
        np.ndarray
            A 1-D NumPy array of 768 features.
        """

        # Load the pre-trained model and processor from HuggingFace
        repo = "microsoft/rad-dino"
        processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo)

        # Preprocess the image and run model inference
        inputs = processor(images=image, return_tensors="pt")

        # Selecting the device to use for torch back-end.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Passing inputs and models to the selected device
        inputs.to(device)
        model.to(device)

        with torch.inference_mode():
            outputs = model(**inputs)

        # Extract and return the CLS embeddings
        cls_embeddings = outputs.pooler_output
        np_embeddings = cls_embeddings.detach().cpu().numpy().squeeze()

        return np_embeddings


class DinoV2FeatureExtractor(FeatureExtractor):

    """
    Feature extractor using models from the HuggingFace DinoV2 family
    (https://huggingface.co/collections/facebook/dinov2-6526c98554b3d2576e071ce3).

    Attributes
    ----------
    model_id : str
        HuggingFace model ID for the selected DinoV2 model.

    Methods
    -------
    feature_extraction(image: PIL.Image.Image) -> np.ndarray
        Extract features from an image using the selected DinoV2 model.

    """

    def __init__(self, model_id: str):
        FeatureExtractor.__init__(self)
        self.model_id = model_id

    def feature_extraction(self, image: PIL.Image) -> np.ndarray:

        """
        Extract features from a PIL image using the selected HuggingFace DinoV2 model.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to extract features from.

        Returns
        -------
        np.ndarray
            A 1-D NumPy array. The number of features depend on the specific model: 384 for small, 768 for base,
            1024 for large, and 1536 for giant.
        """

        processor = AutoImageProcessor.from_pretrained(self.model_id)
        model = AutoModel.from_pretrained(self.model_id)

        inputs = processor(images=image, return_tensors="pt")

        # Selecting the device to use for torch back-end.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Passing inputs and models to the selected device
        inputs.to(device)
        model.to(device)

        with torch.inference_mode():
            outputs = model(**inputs)

        return outputs.pooler_output.detach().cpu().numpy().squeeze()


class MambaFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using models from the HuggingFace MambaVision family.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID for the selected MambaVision model.

    Methods
    -------
    feature_extraction(image: PIL.Image.Image) -> np.ndarray
        Extract features from an image using the selected MambaVision model.
    """

    def __init__(self, model_id: str):
        super().__init__()
        self.model_id = model_id

    def feature_extraction(self, image: PIL.Image.Image) -> np.ndarray:
        """
        Extract features from a PIL image using the selected HuggingFace MambaVision model.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to extract features from.

        Returns
        -------
        np.ndarray
            A 1-D NumPy array of 640 features.
        """

        # Load the specified MambaVision model from HuggingFace
        model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)

        # Switch the model to evaluation mode
        model.cuda().eval()

        # prepare image for the model. Apparently mambavision models requires images with 3 channels.
        input_resolution = (3, image.width, image.height)
        image = image.convert("RGB")

        # Prepare the image using the model's specified resolution and transforms
        transform = timm.data.transforms_factory.create_transform(
            input_size=input_resolution,
            is_training=False,
            mean=model.config.mean,
            std=model.config.std,
            crop_mode=model.config.crop_mode,
            crop_pct=model.config.crop_pct
        )
        inputs = transform(image).unsqueeze(0).cuda()

        # Run inference and return the features
        out_avg_pool, features = model(inputs)

        return out_avg_pool.detach().cpu().numpy().squeeze()