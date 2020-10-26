# %%
from feature_extractor import FeatureExtractor
from normalizer import Normalizer
from reader import PSBDataset, ModelNet40Dataset
from helper.config import FEATURE_DATA_FILE, DEBUG, DATA_PATH_NORMED_SUBSET, DATA_PATH_NORMED, CLASS_FILE, DATA_PATH_PSB, DATA_PATH_DEBUG
import time
import pyvista as pv
import pymeshfix as mf
from pyvista import examples
import numpy as np

# %%
if __name__ == "__main__":
    normalizer = Normalizer(ModelNet40Dataset(DATA_PATH_NORMED))
    normalizer.run_full_pipeline()

# %%
