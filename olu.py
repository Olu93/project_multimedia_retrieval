# %%
from feature_extractor import FeatureExtractor
from normalizer import Normalizer
from reader import PSBDataset
from helper.config import FEATURE_DATA_FILE, DEBUG, DATA_PATH_NORMED_SUBSET, DATA_PATH_NORMED, CLASS_FILE, DATA_PATH_PSB, DATA_PATH_DEBUG
import time

# # %%
# FE = FeatureExtractor(DATA_PATH_NORMED_SUBSET if DEBUG else DATA_PATH_NORMED, FEATURE_DATA_FILE)
# FE.run_full_pipeline()

# %%
# if __name__ == "__main__":
#     norm = Normalizer(PSBDataset(DATA_PATH_DEBUG if DEBUG else DATA_PATH_PSB, class_file_path=CLASS_FILE))
#     norm.run_full_pipeline()

# %%
if __name__ == "__main__":
    dataset = PSBDataset(DATA_PATH_DEBUG, class_file_path=CLASS_FILE)
    norm = Normalizer(dataset)
    FE = FeatureExtractor(DATA_PATH_NORMED_SUBSET)
    # dataset.run_full_pipeline()
    # print(dataset.full_data[0])

# dataset.read()
# # dataset.show_class_histogram()
# dataset.load_files_in_memory()
# dataset.convert_all_to_polydata()
# dataset.compute_shape_statistics()
# dataset.detect_outliers()
# # %%
# dataset.save_statistics("./stats")
# %%
