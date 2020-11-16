# %%
from collections import Counter
import io
import jsonlines
from scipy.spatial.distance import cityblock, cosine, euclidean, sqeuclidean
from scipy.stats.stats import wasserstein_distance
from feature_extractor import FeatureExtractor
from helper.config import FEATURE_DATA_FILE
from evaluator import Evaluator
import pandas as pd
import random
import numpy as np
import csv
import itertools
from tqdm import tqdm


def run_experiment(query_class, function_pipeline, weights, label_coarse):
    evaluator = Evaluator(FEATURE_DATA_FILE, label_coarse=label_coarse)
    mapping = {item["label"]: item["label_coarse"] for item in evaluator.query_matcher.features_raw_init}

    all_data = evaluator.query_matcher.features_raw
    # print(evaluator.mesh_classes_count.keys())
    # print(evaluator.mesh_classes_count[evaluator.mesh_classes_count==10])
    my_subset = [item for item in all_data if item["label"] == (mapping[query_class] if evaluator.label_coarse else query_class)]
    k_all_db_results_cociwa = evaluator.perform_matching_on_subset_slow(my_subset, function_pipeline, weights, k_full_db_switch=True)
    k_all_db_results_cociwa = evaluator.metric_F1(k_all_db_results_cociwa, evaluator.mesh_classes_count, k=10, use_class_k=True)

    k_all_db_results_cociwa[["matches_class"]] = [Counter(tuple(row[["matches_class"]][0])) for index, row in k_all_db_results_cociwa.iterrows()]
    results = k_all_db_results_cociwa[["name", "class", "matches_class"]]
    # print(results.head(10))
    print("")
    results.head(10)
    my_result = evaluator.calc_weighted_metric(k_all_db_results_cociwa)[1]
    print(my_result)
    return my_result


def run_experiment_single(query_mesh, function_pipeline, weights, label_coarse):
    evaluator = Evaluator(FEATURE_DATA_FILE, label_coarse=label_coarse)
    mapping = {item["label"]: item["label_coarse"] for item in evaluator.query_matcher.features_raw_init}

    all_data = evaluator.query_matcher.features_raw
    # print(evaluator.mesh_classes_count.keys())
    # print(evaluator.mesh_classes_count[evaluator.mesh_classes_count==10])
    my_subset = [item for item in all_data if item["name"] == query_mesh]
    k_all_db_results_cociwa = evaluator.perform_matching_on_subset_slow(my_subset, function_pipeline, weights, k_full_db_switch=True)
    k_all_db_results_cociwa = evaluator.metric_F1(k_all_db_results_cociwa, evaluator.mesh_classes_count, k=10, use_class_k=True)

    k_all_db_results_cociwa[["matches_class"]] = [Counter(tuple(row[["matches_class"]][0])) for index, row in k_all_db_results_cociwa.iterrows()]
    results = k_all_db_results_cociwa[["name", "class", "matches_class"]]
    # print(results.head(10))
    print("")
    results.head(10)
    print(evaluator.calc_weighted_metric(k_all_db_results_cociwa)[1])


def run_experiment_single_tmp(query_class, function_pipeline, weights, label_coarse):
    evaluator = Evaluator(FEATURE_DATA_FILE, label_coarse=label_coarse)
    mapping = {item["label"]: item["label_coarse"] for item in evaluator.query_matcher.features_raw_init}

    all_data = evaluator.query_matcher.features_raw
    # print(evaluator.mesh_classes_count.keys())
    # print(evaluator.mesh_classes_count[evaluator.mesh_classes_count==10])
    my_subset = [item for item in all_data if item["label"] == (mapping[query_class] if evaluator.label_coarse else query_class)]
    k_all_db_results_cociwa = evaluator.perform_matching_on_subset_slow(my_subset, function_pipeline, weights, k_full_db_switch=True)
    k_all_db_results_cociwa = evaluator.metric_F1(k_all_db_results_cociwa, evaluator.mesh_classes_count, k=10, use_class_k=True)

    # k_all_db_results_cociwa[["matches_class"]] = [Counter(tuple(row[["matches_class"]][0])) for index, row in k_all_db_results_cociwa.iterrows()]
    # k_all_db_results_cociwa[["matches_class"]] = [Counter(tuple(row[["matches_class"]][0])) for index, row in k_all_db_results_cociwa.iterrows()]
    results = k_all_db_results_cociwa[["name", "class", "matches"]]
    # print(results.head(10))
    # print("")
    results.head(10)
    # print(k_all_db_results_cociwa)
    return k_all_db_results_cociwa[k_all_db_results_cociwa["class"] == query_class][["name", "class", "matches", "matches_class"]]


# %%
if __name__ == "__main__":
    features_df_raw = pd.DataFrame([data for data in jsonlines.Reader(io.open(FEATURE_DATA_FILE))])
    count_hists = sum([1 for header_name in features_df_raw.columns if "hist_" in header_name])
    count_skeletons = sum([1 for header_name in features_df_raw.columns if "skeleton_" in header_name])
    weights_wo_skeleton = ([1]) + ([1] * count_hists) + ([0] * count_skeletons)
    weights_wo_hist = ([1]) + ([0] * count_hists) + ([1] * count_skeletons)
    weights_wo_scalar = ([0]) + ([1] * count_hists) + ([1] * count_skeletons)
    weights = ([1]) + ([1] * count_hists) + ([1] * count_skeletons)
    weights_w_strong_wo_skeleton = ([1]) + ([100] * count_hists) + ([0] * count_skeletons)
    weights_w_strong = ([3]) + ([100] * count_hists) + ([1] * count_skeletons)
    function_pipeline = [cosine] + ([wasserstein_distance] * count_hists) + ([euclidean] * count_skeletons)

    # variables = run_experiment_single_tmp("ant", function_pipeline, weights_w_strong, False)

    # function_pipeline = [cosine] + ([wasserstein_distance] * count_hists) + ([cityblock] * count_skeletons)
    # %%
    lst = list(itertools.product([0, 1], repeat=len(weights)))
    all_results = {}
    labels_to_test = "ant,helmet,desk_lamp".split(",")
    cols = [f"w_{idx}" for idx in range(len(weights))] + ["class_label", "val"]


    with io.open("feature_combinations.csv", "w", newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, cols)
        writer.writeheader()
        for idx in tqdm(list(range(4000))):
            print(f"===" * 10)
            lst_sample = random.sample(lst, k=1)[0]
            print(f"Round: {idx}")
            print(f"Current selected combination is: {lst_sample}")
            if lst_sample in all_results:
                print(f"Skip: {lst_sample}")
                continue

            weights_w_strong = lst_sample
            for class_label in labels_to_test:
                value = run_experiment(class_label, function_pipeline, weights_w_strong, False)
                all_results[lst_sample + (class_label, )] = value
                writer.writerow(dict(**{f"w_{idx}":val for idx, val in enumerate(lst_sample)}, class_label=class_label, val=value))

    # all_results_df = pd.DataFrame([combi + (val,) for combi, val in all_results.items()], columns=cols)
    all_results_df = pd.read_csv("feature_combinations.csv")
    # %%

#     for idx in range(10):
#         print(f"============== {idx} ==============")
#         weights_w_strong = ([3]) + ([100] * count_hists) + ([1] * count_skeletons)
#         run_experiment("ant", function_pipeline, weights_w_strong, False)

#     # %%
#     for idx in range(5):
#         print(f"============= {idx} ===============")
#         weights_w_strong = ([idx]) + ([100] * count_hists) + ([0] * count_skeletons)
#         run_experiment("helmet", function_pipeline, weights_w_strong, False)
#         weights_w_strong = ([idx]) + ([100] * count_hists) + ([1] * count_skeletons)
#         run_experiment("helmet", function_pipeline, weights_w_strong, False)

#     # %%
#     for idx in range(5):
#         print(f"============= {idx} ===============")
#         weights_w_strong = ([idx]) + ([100] * count_hists) + ([0] * count_skeletons)
#         run_experiment("bridge", function_pipeline, weights_w_strong, False)
#         weights_w_strong = ([idx]) + ([100] * count_hists) + ([1] * count_skeletons)
#         run_experiment("bridge", function_pipeline, weights_w_strong, False)

#     # run_experiment("helmet", function_pipeline, weights, False)

# #     # %%
# #     run_experiment("helmet", function_pipeline, weights, False)
# #     run_experiment("helmet", function_pipeline, weights, True)
# #     print("================================================")
# #     run_experiment("ant", function_pipeline, weights, False)
# #     run_experiment("ant", function_pipeline, weights, True)
# #     print("================================================")
# #     run_experiment("bridge", function_pipeline, weights, False)
# #     run_experiment("bridge", function_pipeline, weights, True)

# # %%
#     print("Skeleton features experiment!")
#     run_experiment("helmet", function_pipeline, weights_wo_skeleton, False)
#     run_experiment("helmet", function_pipeline, weights, False)
#     print("================================================")
#     run_experiment("ant", function_pipeline, weights_wo_skeleton, False)
#     run_experiment("ant", function_pipeline, weights, False)
#     print("================================================")
#     run_experiment("bridge", function_pipeline, weights_wo_skeleton, False)
#     run_experiment("bridge", function_pipeline, weights, False)
#     # %%
#     print("Hist features experiment!")
#     run_experiment("helmet", function_pipeline, weights_wo_hist, False)
#     run_experiment("helmet", function_pipeline, weights, False)
#     print("================================================")
#     run_experiment("ant", function_pipeline, weights_wo_hist, False)
#     run_experiment("ant", function_pipeline, weights, False)
#     print("================================================")
#     run_experiment("bridge", function_pipeline, weights_wo_hist, False)
#     run_experiment("bridge", function_pipeline, weights, False)

#     # %%
#     print("Scalar features experiment!")
#     run_experiment("helmet", function_pipeline, weights_wo_scalar, False)
#     run_experiment("helmet", function_pipeline, weights, False)
#     print("================================================")
#     run_experiment("ant", function_pipeline, weights_wo_scalar, False)
#     run_experiment("ant", function_pipeline, weights, False)
#     print("================================================")
#     run_experiment("bridge", function_pipeline, weights_wo_scalar, False)
#     run_experiment("bridge", function_pipeline, weights, False)

#     # %%
#     print("Hist features stronger weights experiment!")
#     run_experiment("helmet", function_pipeline, weights, False)
#     run_experiment("helmet", function_pipeline, weights_w_strong_hist, False)
#     print("================================================")
#     run_experiment("ant", function_pipeline, weights, False)
#     run_experiment("ant", function_pipeline, weights_w_strong_hist, False)
#     print("================================================")
#     run_experiment("bridge", function_pipeline, weights, False)
#     run_experiment("bridge", function_pipeline, weights_w_strong_hist, False)
# # %%

# # %%
# - Values improve for scalar weights 3
# - Values improve for scalar weights 3
