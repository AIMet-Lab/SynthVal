import synthval.feature_extraction

source_folder_path = ""

feature_extractors = [synthval.feature_extraction.RadDinoFeatureExtractor(),
                      synthval.feature_extraction.DinoV2FeatureExtractor("facebook/dinov2-base"),
                      synthval.feature_extraction.MambaFeatureExtractor("nvidia/MambaVision-B-1K")]

save_paths = ["csaw_rad-dino_features.csv",
              "csaw_dinov2-base_features.csv",
              "csaw_mamba-base_features.csv"]

for i in range(len(feature_extractors)):
    feature_extractor = feature_extractors[i]
    save_path = save_paths[i]
    feature_extractor.get_features_df(source_folder_path=source_folder_path, save_path=save_path)