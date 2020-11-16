# %%
from collections import Counter
import io
import jsonlines
from scipy.spatial.distance import cityblock, cosine, euclidean
from scipy.stats.stats import wasserstein_distance
from tqdm.std import tqdm
from feature_extractor import FeatureExtractor
from helper.config import FEATURE_DATA_FILE
from evaluator import Evaluator
import pandas as pd
import itertools
import numpy as np
import csv


def calc_F1(evaluator, function_pipeline, weights) -> pd.DataFrame:
    k_all_db_results = evaluator.perform_matching_calculate_metrics(function_pipeline, weights, k_full_db_switch=True)
    return evaluator.metric_F1(k_all_db_results, evaluator.mesh_classes_count, k=10, use_class_k=False, weightbool=True, weight=0.25)


if __name__ == "__main__":
    evaluator = Evaluator(FEATURE_DATA_FILE, label_coarse=False)
    features_df_raw = pd.DataFrame([data for data in jsonlines.Reader(io.open(FEATURE_DATA_FILE))])
    count_hists = sum([1 for header_name in features_df_raw.columns if "hist_" in header_name])
    count_skeletons = sum([1 for header_name in features_df_raw.columns if "skeleton_" in header_name])

    weights_all_one = ([None]) + ([None] * count_hists) + ([None] * count_skeletons)
    function_pipeline = [cosine] + ([wasserstein_distance] * count_hists) + ([euclidean] * count_skeletons)
    metrics = calc_F1(evaluator, function_pipeline, weights_all_one)
    metrics.to_csv("stats/final_evaluation.csv", index=False)

    #Erweitern oder dir fällt ne schickere Möglichkeit ein alle Kombinationen zu versammeln

    # Chris script dazu
