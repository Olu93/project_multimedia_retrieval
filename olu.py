from feature_extractor import FeatureExtractor
from helper.config import FEATURE_DATA_FILE, DEBUG, DATA_PATH_NORMED_SUBSET, DATA_PATH_NORMED
import time
if __name__ == "__main__":
    FE = FeatureExtractor(DATA_PATH_NORMED_SUBSET if DEBUG else DATA_PATH_NORMED, FEATURE_DATA_FILE)
    FE.run_full_pipeline()