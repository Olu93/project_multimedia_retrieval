# %%
from collections import Counter
import io
import jsonlines
from scipy.spatial.distance import cityblock, cosine, euclidean, sqeuclidean
from scipy.stats.stats import wasserstein_distance
from helper.config import FEATURE_DATA_FILE
from evaluator import Evaluator
import pandas as pd
import random
import numpy as np
import csv
from tqdm import tqdm
import multiprocessing as mp




def run_experiment(query_class, function_pipeline, weights, label_coarse):
    evaluator = Evaluator(FEATURE_DATA_FILE, label_coarse=label_coarse)
    mapping = {item["label"]: item["label_coarse"] for item in evaluator.query_matcher.features_raw_init}

    all_data = evaluator.query_matcher.features_raw
    my_subset = [item for item in all_data if item["label"] == (mapping[query_class] if evaluator.label_coarse else query_class)]
    k_all_db_results_cociwa = evaluator.perform_matching_on_subset_slow(my_subset, function_pipeline, weights, k_full_db_switch=True)
    k_all_db_results_cociwa = evaluator.metric_F1(k_all_db_results_cociwa, evaluator.mesh_classes_count, k=10, use_class_k=True, silent=True)

    k_all_db_results_cociwa[["matches_class"]] = [Counter(tuple(row[["matches_class"]][0])) for index, row in k_all_db_results_cociwa.iterrows()]
    results = k_all_db_results_cociwa[["name", "class", "matches_class"]]
    results.head(10)
    my_result = evaluator.calc_weighted_metric(k_all_db_results_cociwa, "F1score", "F1")[1]
    return my_result


def run_experiment_single(query_mesh, function_pipeline, weights, label_coarse):
    evaluator = Evaluator(FEATURE_DATA_FILE, label_coarse=label_coarse)
    mapping = {item["label"]: item["label_coarse"] for item in evaluator.query_matcher.features_raw_init}
    all_data = evaluator.query_matcher.features_raw
    my_subset = [item for item in all_data if item["name"] == query_mesh]
    k_all_db_results_cociwa = evaluator.perform_matching_on_subset_slow(my_subset, function_pipeline, weights, k_full_db_switch=True)
    k_all_db_results_cociwa = evaluator.metric_F1(k_all_db_results_cociwa, evaluator.mesh_classes_count, k=10, use_class_k=True)
    k_all_db_results_cociwa[["matches_class"]] = [Counter(tuple(row[["matches_class"]][0])) for index, row in k_all_db_results_cociwa.iterrows()]
    results = k_all_db_results_cociwa[["name", "class", "matches_class"]]
    print("")
    results.head(10)
    print(evaluator.calc_weighted_metric(k_all_db_results_cociwa)[1])


def run_experiment_single_tmp(query_class, function_pipeline, weights, label_coarse):
    evaluator = Evaluator(FEATURE_DATA_FILE, label_coarse=label_coarse)
    mapping = {item["label"]: item["label_coarse"] for item in evaluator.query_matcher.features_raw_init}
    all_data = evaluator.query_matcher.features_raw
    my_subset = [item for item in all_data if item["label"] == (mapping[query_class] if evaluator.label_coarse else query_class)]
    k_all_db_results_cociwa = evaluator.perform_matching_on_subset_slow(my_subset, function_pipeline, weights, k_full_db_switch=True)
    k_all_db_results_cociwa = evaluator.metric_F1(k_all_db_results_cociwa, evaluator.mesh_classes_count, k=10, use_class_k=True)
    results = k_all_db_results_cociwa[["name", "class", "matches"]]
    results.head(10)
    return k_all_db_results_cociwa[k_all_db_results_cociwa["class"] == query_class][["name", "class", "matches", "matches_class"]]


def parallel_run_experiment(combi):
    sr, hr, skr, class_label, function_pipeline, count_hists, count_skeletons = combi
    print(f"Current selected combination is: {combi[:4]}")
    weights_w_strong = ([sr]) + ([hr] * count_hists) + ([skr] * count_skeletons)
    value = run_experiment(class_label, function_pipeline, weights_w_strong, False)
    return {"sr": sr, "hr": hr, "skr": skr, "class_label": class_label, "val": value}


if __name__ == "__main__":
    features_df_raw = pd.DataFrame([data for data in jsonlines.Reader(io.open(FEATURE_DATA_FILE))])
    count_hists = sum([1 for header_name in features_df_raw.columns if "hist_" in header_name])
    count_skeletons = sum([1 for header_name in features_df_raw.columns if "skeleton_" in header_name])
    function_pipeline = [cosine] + ([wasserstein_distance] * count_hists) + ([cityblock] * count_skeletons)
    scalar_range = np.linspace(1, 10, 20)
    hist_range = np.linspace(1, 300, 20)
    skeleton_range = np.linspace(1, 10, 20)
    experiment_params = []
    labels_to_test = "ant,helmet,desk_lamp,bench,tv,bottle,acoustic_guitar".split(",")
    cols = "sr hr skr class_label val".split()
    pool = mp.Pool(6)

    with io.open("stats/hyper_params_informed_func_combo.csv", "w", newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, cols)
        writer.writeheader()
        for idx in tqdm(range(1000), total=1000):
            sr = np.random.choice(scalar_range).round(2)
            hr = np.random.choice(hist_range).round(2)
            skr = np.random.choice(skeleton_range).round(2)
            combi = (sr, hr, skr)
            for class_label in labels_to_test:
                experiment_params.append((sr, hr, skr, class_label, tuple(function_pipeline), count_hists, count_skeletons))
        unique_configurations = set(experiment_params)

        for item in pool.imap(parallel_run_experiment, tqdm(unique_configurations, total=len(unique_configurations)), chunksize=10):
            writer.writerow(item)
            print(f"Wrote: {item}")

    # all_results_df = pd.DataFrame([combi + (val,) for combi, val in all_results.items()], columns=cols)
    # all_results_df = pd.read_csv("stats/hyper_params.csv")

    #     for idx in range(10):
    #         print(f"============== {idx} ==============")
    #         weights_w_strong = ([3]) + ([100] * count_hists) + ([1] * count_skeletons)
    #         run_experiment("ant", function_pipeline, weights_w_strong, False)

    #     # %%

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

    # best_weight_combos_data = pd.read_csv('feature_combinations.csv')

    # def get_top_weights(df, class_label):
    #     top_ant_val = df[df.class_label == class_label].val.max()
    #     top_features = df[df.val > .90 * top_ant_val]
    #     top_features_counts = top_features.sum()
    #     top_weights = top_features_counts.iloc[:-2].values
    #     weights_w_strong = ([2.9]) + ([134] * count_hists) + ([1.58] * count_skeletons)
    #     return tuple(top_weights * weights_w_strong)

    # top_ant_weights = get_top_weights(best_weight_combos_data, "ant")
    # top_helmet_weights = get_top_weights(best_weight_combos_data, "helmet")

    # print(f"===" * 10)
    # run_experiment("ant", function_pipeline, top_ant_weights, False)
    # run_experiment("helmet", function_pipeline, top_ant_weights, False)
    # run_experiment("bridge", function_pipeline, top_ant_weights, False)
    # print(f"===" * 10)
    # run_experiment("ant", function_pipeline, top_helmet_weights, False)
    # run_experiment("helmet", function_pipeline, top_helmet_weights, False)
    # run_experiment("bridge", function_pipeline, top_helmet_weights, False)

    # # %%
    # run_experiment("ant", function_pipeline, top_ant_weights, False)
    # run_experiment("spider", function_pipeline, top_ant_weights, False)

    # # %%
    # result1 = run_experiment_single_tmp("ant", function_pipeline, top_ant_weights, False)
    # result2 = run_experiment_single_tmp("ant", function_pipeline, weights_w_strong, False)