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


def calc_F1(evaluator, function_pipeline, weights):
    k_all_db_results = evaluator.perform_matching_calculate_metrics(function_pipeline, weights, k_full_db_switch=True)
    k_all_db_results = evaluator.metric_F1(k_all_db_results, evaluator.mesh_classes_count, k=10, use_class_k=False, weightbool=True, weight=0.25)
    F1 = evaluator.calc_weighted_metric(k_all_db_results, "F1score", "F1")[1]
    print(F1)
    return F1


if __name__ == "__main__":
    evaluator = Evaluator(FEATURE_DATA_FILE, label_coarse=False)
    features_df_raw = pd.DataFrame([data for data in jsonlines.Reader(io.open(FEATURE_DATA_FILE))])
    count_hists = sum([1 for header_name in features_df_raw.columns if "hist_" in header_name])
    count_skeletons = sum([1 for header_name in features_df_raw.columns if "skeleton_" in header_name])
    weights_all_one = ([1]) + ([1] * count_hists) + ([1] * count_skeletons)

    #Erweitern oder dir fällt ne schickere Möglichkeit ein alle Kombinationen zu versammeln
    function_set = [[euclidean, cityblock, wasserstein_distance, cosine]] * 3
    func_combs = list(itertools.product(*function_set))

    F1_scores = []
    pre_comp_data = pd.read_csv("stats/func_combos.csv", index_col=None)
    with io.open("stats/func_combos.csv", "a", newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, "f1 f2 f3 F1".split())
        writer.writeheader()
        for f1, f2, f3 in tqdm(func_combs, total=len(func_combs)):
            combi = f"{f1.__name__[:2]}{f2.__name__[:2]}{f3.__name__[:2]}".upper()
            if ((pre_comp_data['f1'] == f1.__name__) & (pre_comp_data['f2'] == f2.__name__) & (pre_comp_data['f3'] == f3.__name__)).any():
                print(f"Skip {combi} - Already computed!")
                continue
            print(f"===" * 10)
            print(f"Current selected combination is: {combi}")
            funcs = [f1] + ([f2] * count_hists) + ([f3] * count_skeletons)
            F1 = calc_F1(evaluator, funcs, weights_all_one)
            writer.writerow(dict(f1=f1.__name__, f2=f2.__name__, f3=f3.__name__, F1=F1))
            csv_file.flush()
