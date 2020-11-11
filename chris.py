# %%
from collections import Counter
import io
import jsonlines
from scipy.spatial.distance import cityblock, cosine, euclidean
from scipy.stats.stats import wasserstein_distance
from feature_extractor import FeatureExtractor
from helper.config import FEATURE_DATA_FILE
from evaluator import Evaluator
import pandas as pd
import itertools
import numpy as np

def calc_F1(function_pipeline, weights):
    evaluator = Evaluator(FEATURE_DATA_FILE, label_coarse=False)
    k_all_db_results = evaluator.perform_matching_calculate_metrics(function_pipeline, weights, k_full_db_switch=True)
    k_all_db_results = evaluator.metric_F1(k_all_db_results, evaluator.mesh_classes_count, k=10, use_class_k=False, weightbool=True, weight=0.25)
    F1 = evaluator.calc_weighted_metric(k_all_db_results, "F1score", "F1")[1]
    print(F1)
    return F1

# %%
if __name__ == "__main__":
    features_df_raw = pd.DataFrame([data for data in jsonlines.Reader(io.open(FEATURE_DATA_FILE))])
    count_hists = sum([1 for header_name in features_df_raw.columns if "hist_" in header_name])
    count_skeletons = sum([1 for header_name in features_df_raw.columns if "skeleton_" in header_name])
    weights_all_one = ([1]) + ([1] * count_hists) + ([1] * count_skeletons)

    #Erweitern oder dir fällt ne schickere Möglichkeit ein alle Kombinationen zu versammeln
    func_combs = [
        ["EUCIWA"] + [euclidean] + ([cityblock] * count_hists) + ([wasserstein_distance] * count_skeletons),
        ["EUCICI"] + [euclidean] + ([cityblock] * count_hists) + ([cityblock] * count_skeletons),
        ["EUCIEU"] + [euclidean] + ([cityblock] * count_hists) + ([euclidean] * count_skeletons),
        ["EUWAWA"] + [euclidean] + ([wasserstein_distance] * count_hists) + ([wasserstein_distance] * count_skeletons),
        ["EUWACI"] + [euclidean] + ([wasserstein_distance] * count_hists) + ([cityblock] * count_skeletons),
        ["EUWAEU"] + [euclidean] + ([wasserstein_distance] * count_hists) + ([euclidean] * count_skeletons),

        ["COCIWA"] + [cosine] + ([cityblock] * count_hists) + ([wasserstein_distance] * count_skeletons),
        ["COCICI"] + [cosine] + ([cityblock] * count_hists) + ([cityblock] * count_skeletons),
        ["COCIEU"] + [cosine] + ([cityblock] * count_hists) + ([euclidean] * count_skeletons),
        ["COWAWA"] + [cosine] + ([wasserstein_distance] * count_hists) + ([wasserstein_distance] * count_skeletons),
        ["COWACI"] + [cosine] + ([wasserstein_distance] * count_hists) + ([cityblock] * count_skeletons),
        ["COWAEU"] + [cosine] + ([wasserstein_distance] * count_hists) + ([euclidean] * count_skeletons)
    ]

    F1_scores = []
    for pipeline in func_combs:
        combi = pipeline[0]
        print(combi)
        F1 = calc_F1(pipeline[1:], weights_all_one)
        F1_scores.append([combi, F1])


