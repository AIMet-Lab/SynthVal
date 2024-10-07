import synthval.feature_extraction

extractor = synthval.feature_extraction.DinoV2FeatureExtractor("facebook/dinov2-base")
df = extractor.get_features_df(source_folder_path="/Users/dguidotti/Documents/Git Projects/SynthVal/test_images/")
