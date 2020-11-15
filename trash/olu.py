# %%
from collections import Counter
import io
from normalizer import Normalizer
import jsonlines
from scipy.spatial.distance import cityblock, cosine, euclidean, sqeuclidean
from scipy.stats.stats import wasserstein_distance
from feature_extractor import FeatureExtractor
from helper.config import DATA_PATH_MN40, DATA_PATH_MN40_SUBSET, DEBUG, FEATURE_DATA_FILE, DATA_PATH_NORMED_SUBSET
from evaluator import Evaluator
import pandas as pd
import random
import numpy as np
import csv
from tqdm import tqdm
from reader import ModelNet40Dataset


if __name__ == '__main__':
    norm = Normalizer(ModelNet40Dataset(DATA_PATH_MN40))
    norm.run_full_pipeline(10 if DEBUG else None)
    print("Done")

# from tqdm import tqdm
# from glob import glob
# import io
# import os

# for file_name in tqdm(glob("data/mn40/sofa/**/*.off", recursive=True)):
#     if "_modified.off" not in file_name:
#         print(file_name)

#         os.remove(file_name)
#     # file_reader = io.open(file_name, "r")
#     # content = file_reader.readlines()
#     # if content[0].strip() != "OFF" and not "_mod.off" in file_name:

#     #     off_part, rest = content[0][:3], content[0][3:]
#     #     file_writer = io.open(f"{file_name[:-4]}_modified.off", "w")
#     #     file_writer.write(f"{off_part}{file_reader.newlines}")
#     #     file_writer.write(f"{rest}")
#     #     file_writer.writelines(content[1:])
#     #     print(f"Wrote file:{file_writer.name}")
