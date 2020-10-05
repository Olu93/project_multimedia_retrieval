from feature_extractor import FeatureExtractor
from helper.config import FEATURE_DATA_FILE
import time
if __name__ == "__main__":
    FE = FeatureExtractor(FEATURE_DATA_FILE)
    FE.run_full_pipeline(10)