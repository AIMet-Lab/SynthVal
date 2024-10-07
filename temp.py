import synthval.feature_extraction

extractor = synthval.feature_extraction.RadDinoFeatureExtractor()
df = extractor.get_features_df(source_folder_path="/Users/dguidotti/Documents/Git Projects/SynthVal/test_images/")
