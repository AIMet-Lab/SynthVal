import synthval.features_extraction

if __name__ == "__main__":

    # === Configuration / parameters ===
    real_images_folder = "data/images/real/"
    synth_images_folder = "data/images/synth/"
    features_extractor_id = "facebook/dinov2-small"
    real_features_df_savepath = "data/features/real_dinov2-small.csv"
    synth_features_df_savepath = "data/features/synth_dinov2-small.csv"

    # Instantiate a feature extractor using the DINOv2 model
    # The class is part of synthval.features_extraction
    features_extractor = synthval.features_extraction.DinoV2FeatureExtractor(
        model_id = features_extractor_id
    )

    # Run feature extraction on the real images folder
    # and save the results to a CSV file
    features_extractor.get_features_df(
        source_folder_path = real_images_folder,
        save_path = real_features_df_savepath
    )

    # Similarly, run feature extraction on the synthetic images folder
    # and save to another CSV file
    features_extractor.get_features_df(
        source_folder_path = synth_images_folder,
        save_path = synth_features_df_savepath
    )
