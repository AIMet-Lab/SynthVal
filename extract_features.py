
import synthval.feature_extraction
import os

master_folder_path = "/run/user/1000/gvfs/smb-share:server=wdmycloudex4100.local,share=public/CSAW Dataset/2021-204-1-1/"
categories = ["L_MLO", "R_MLO", "L_CC", "R_CC"]
out_folder_path = "temp_out/"

feature_extractors = [synthval.feature_extraction.RadDinoFeatureExtractor(),
                      synthval.feature_extraction.DinoV2FeatureExtractor("facebook/dinov2-base"),
                      synthval.feature_extraction.MambaFeatureExtractor("nvidia/MambaVision-B-1K"),
                      synthval.feature_extraction.DinoV2FeatureExtractor("facebook/dinov2-small"),
                      synthval.feature_extraction.MambaFeatureExtractor("nvidia/MambaVision-S-1K"),
                      synthval.feature_extraction.DinoV2FeatureExtractor("facebook/dinov2-large"),
                      synthval.feature_extraction.MambaFeatureExtractor("nvidia/MambaVision-L-1K"),
                      synthval.feature_extraction.DinoV2FeatureExtractor("facebook/dinov2-giant"),
                      synthval.feature_extraction.MambaFeatureExtractor("nvidia/MambaVision-L2-1K"),
                      ]

models_ids = ["rad-dino",
              "dinov2-base",
              "mamba-base"
              "dinov2-small",
              "mamba-small"
              "dinov2-large",
              "mamba-large"
              "dinov2-giant",
              "mamba-giant"]

for i in range(len(feature_extractors)):
    feature_extractor = feature_extractors[i]
    model_id = models_ids[i]
    for cat in categories:
        save_path = os.path.join("temp_output/", f"CSAW_{cat}_{model_id}_features.csv")
        source_folder_path = os.path.join(master_folder_path, f"{cat}/")
        feature_extractor.get_features_df(source_folder_path=source_folder_path, save_path=save_path)