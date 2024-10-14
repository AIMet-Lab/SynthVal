
import synthval.features_extraction
import torch


def test_rad_dino_fe():

    image_folder_path = "test_data/images/"
    feature_extractor = synthval.features_extraction.RadDinoFeatureExtractor()
    feature_extractor.get_features_df(image_folder_path, verbose=False)


def test_dino_fe():

    image_folder_path = "test_data/images/"
    feature_extractor = synthval.features_extraction.DinoV2FeatureExtractor("facebook/dinov2-small")
    feature_extractor.get_features_df(image_folder_path, verbose=False)


def test_mamba_fe():

    image_folder_path = "test_data/images/"
    feature_extractor = synthval.features_extraction.DinoV2FeatureExtractor("nvidia/MambaVision-T-1K")
    feature_extractor.get_features_df(image_folder_path, verbose=False)


def test_inception_fe():
    image_folder_path = "test_data/images/"
    feature_extractor = synthval.features_extraction.InceptionExtractor("inception_v3")
    feature_extractor.get_features_df(image_folder_path, verbose=False)
    feature_extractor = synthval.features_extraction.InceptionExtractor("inception_v3", get_probabilities=True)
    feature_extractor.get_features_df(image_folder_path, verbose=False)


if __name__ == '__main__':

    test_rad_dino_fe()
    test_dino_fe()
    if torch.cuda.is_available():
        test_mamba_fe()
    test_inception_fe()
