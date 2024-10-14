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

import PIL.Image
import pandas
import synthval

import numpy
import torch
import transformers


import timm
import timm.data
import timm.data.transforms_factory


class FeatureExtractor(abc.ABC):
    """
    Abstract base class representing a generic feature extractor. Child classes must implement
    the concrete `feature_extraction` method for specific feature extraction.

    """

    @abc.abstractmethod
    def feature_extraction(self, image: PIL.Image.Image) -> numpy.ndarray:
        """
        Abstract method to extract features from a PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
            The image from which features are to be extracted.

        Returns
        -------
        numpy.ndarray
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
            stream_logger = synthval.utilities.get_stream_logger("synthval.feature_extraction")

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
            pil_image = synthval.utilities.get_pil_image(image_path)

            # Extract features from the image
            np_features = self.feature_extraction(pil_image)

            # Append the features to the dataset
            features_dataset.append(np_features)

        # Convert the list of features to a NumPy array and then to a DataFrame
        features_dataset = numpy.array(features_dataset).squeeze()
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

    """

    def __init__(self):
        super().__init__()

    def feature_extraction(self, image: PIL.Image.Image) -> numpy.ndarray:
        """
        Extract features from a PIL image using the HuggingFace Rad-Dino model.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to extract features from.

        Returns
        -------
        numpy.ndarray
            A 1-D NumPy array of 768 features.
        """

        # Load the pre-trained model and processor from HuggingFace
        repo = "microsoft/rad-dino"
        processor = transformers.AutoImageProcessor.from_pretrained(repo)
        model = transformers.AutoModel.from_pretrained(repo)

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

    """

    def __init__(self, model_id: str):
        FeatureExtractor.__init__(self)
        self.model_id = model_id

    def feature_extraction(self, image: PIL.Image.Image) -> numpy.ndarray:

        """
        Extract features from a PIL image using the selected HuggingFace DinoV2 model.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to extract features from.

        Returns
        -------
        numpy.ndarray
            A 1-D NumPy array. The number of features depend on the specific model: 384 for small, 768 for base,
            1024 for large, and 1536 for giant.
        """

        processor = transformers.AutoImageProcessor.from_pretrained(self.model_id)
        model = transformers.AutoModel.from_pretrained(self.model_id)

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

    """

    def __init__(self, model_id: str):
        super().__init__()
        self.model_id = model_id

    def feature_extraction(self, image: PIL.Image.Image) -> numpy.ndarray:
        """
        Extract features from a PIL image using the selected HuggingFace MambaVision model.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to extract features from.

        Returns
        -------
        numpy.ndarray
            A 1-D NumPy array of 640 features.
        """

        # Load the specified MambaVision model from HuggingFace
        model = transformers.AutoModel.from_pretrained(self.model_id, trust_remote_code=True)

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


class InceptionExtractor(FeatureExtractor):
    """
    Feature extractor using traditional Inception models from the timm library (https://huggingface.co/docs/timm/index).

    Attributes
    ----------
    model_id : str
        Timm model ID for the selected Inception model (should be one among inception_v3 and inception_v4).

    get_probabilities : bool, Optional
        Flag controlling if the extractor should provide the model prediction over the
        ImageNet (https://www.image-net.org) classes. We provide this capability for the computation of the
        Inception Score (default: False).

    """

    def __init__(self, model_id: str, get_probabilities: bool = False):
        FeatureExtractor.__init__(self)
        self.model_id = model_id
        self.get_probabilities = get_probabilities

    def feature_extraction(self, image: PIL.Image.Image) -> numpy.ndarray:
        """
        Extract features (or probabilities if get_probabilities is set) from a PIL image using the selected
        Timm Inception model.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to extract features from.

        Returns
        -------
        numpy.ndarray
            A 1-D NumPy array of 2048 features (or probabilities).
        """

        # Load the specified MambaVision model from HuggingFace

        if self.get_probabilities:
            model = timm.create_model(self.model_id, pretrained=True)
        else:
            model = timm.create_model(self.model_id, pretrained=True, num_classes=0)

        # Switch the model to evaluation mode
        model.eval()

        # Prepare image for the model and convert to tensor. Inception models expects images in RGB.
        image = image.convert('RGB')

        # Prepare the image using the model's specified resolution and transforms
        config = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.transforms_factory.create_transform(**config)

        inputs = transform(image).unsqueeze(0)

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

        with torch.no_grad():
            out = model(inputs)

        if self.get_probabilities:
            out = torch.nn.functional.softmax(out[0], dim=0)

        return out.detach().cpu().numpy().squeeze()
