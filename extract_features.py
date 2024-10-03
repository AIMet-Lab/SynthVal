import synthval.feature_extraction
import os


master_folder_path = "smb://wdmycloudex4100.local/public/CSAW%20Dataset/2021-204-1-1/"
categories = ["L_MLO", "R_MLO", "L_CC", "R_CC"]
out_folder_path = "temp_out/"

feature_extractors = [synthval.feature_extraction.RadDinoFeatureExtractor(),
                      synthval.feature_extraction.DinoV2FeatureExtractor("facebook/dinov2-base"),
                      synthval.feature_extraction.MambaFeatureExtractor("nvidia/MambaVision-B-1K")]

fe_ids = ["rad-dino",
          "dinov2-base",
          "mamba-base"]

for i in range(len(feature_extractors)):
    feature_extractor = feature_extractors[i]
    fe_id = fe_ids[i]
    for cat in categories:
        source_folder_path = os.path.join(master_folder_path, f"{cat}/")
        save_path = os.path.join(out_folder_path, f"csaw_{cat}_{fe_id}_features.csv")
        feature_extractor.get_features_df(source_folder_path=source_folder_path,
                                          save_path=save_path)