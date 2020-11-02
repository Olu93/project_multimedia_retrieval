# %%
from collections import Counter
import io
import jsonlines
from scipy.spatial.distance import cityblock, cosine
from scipy.stats.stats import wasserstein_distance
from feature_extractor import FeatureExtractor
from helper.config import FEATURE_DATA_FILE
from evaluator import Evaluator
import pandas as pd


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
    print(evaluator.calc_weighted_metric(k_all_db_results_cociwa)[1])

# %%
if __name__ == "__main__":
    features_df_raw = pd.DataFrame([data for data in jsonlines.Reader(io.open(FEATURE_DATA_FILE))])
    count_hists = sum([1 for header_name in features_df_raw.columns if "hist_" in header_name])
    count_skeletons = sum([1 for header_name in features_df_raw.columns if "skeleton_" in header_name])
    weights_wo_skeleton = ([1]) + ([1] * count_hists) + ([0] * count_skeletons)
    weights_wo_hist = ([1]) + ([0] * count_hists) + ([1] * count_skeletons)
    weights_wo_scalar = ([0]) + ([1] * count_hists) + ([1] * count_skeletons)
    weights = ([1]) + ([1] * count_hists) + ([1] * count_skeletons)
    weights_w_strong_hist = ([2]) + ([1] * count_hists) + ([1] * count_skeletons)
    function_pipeline = [cosine] + ([wasserstein_distance] * count_hists) + ([cityblock] * count_skeletons)
    # function_pipeline = [cosine] + ([wasserstein_distance] * count_hists) + ([cityblock] * count_skeletons)

    # %%
    run_experiment("helmet", function_pipeline, weights, False)
    run_experiment("helmet", function_pipeline, weights, True)
    print("================================================")
    run_experiment("ant", function_pipeline, weights, False)
    run_experiment("ant", function_pipeline, weights, True)
    print("================================================")
    run_experiment("bridge", function_pipeline, weights, False)
    run_experiment("bridge", function_pipeline, weights, True)

    # %%
    print("Skeleton features experiment!")
    run_experiment("helmet", function_pipeline, weights_wo_skeleton, False)
    run_experiment("helmet", function_pipeline, weights, False)
    print("================================================")
    run_experiment("ant", function_pipeline, weights_wo_skeleton, False)
    run_experiment("ant", function_pipeline, weights, False)
    print("================================================")
    run_experiment("bridge", function_pipeline, weights_wo_skeleton, False)
    run_experiment("bridge", function_pipeline, weights, False)
    # %%
    print("Hist features experiment!")
    run_experiment("helmet", function_pipeline, weights_wo_hist, False)
    run_experiment("helmet", function_pipeline, weights, False)
    print("================================================")
    run_experiment("ant", function_pipeline, weights_wo_hist, False)
    run_experiment("ant", function_pipeline, weights, False)
    print("================================================")
    run_experiment("bridge", function_pipeline, weights_wo_hist, False)
    run_experiment("bridge", function_pipeline, weights, False)

    # %%
    print("Scalar features experiment!")
    run_experiment("helmet", function_pipeline, weights_wo_scalar, False)
    run_experiment("helmet", function_pipeline, weights, False)
    print("================================================")
    run_experiment("ant", function_pipeline, weights_wo_scalar, False)
    run_experiment("ant", function_pipeline, weights, False)
    print("================================================")
    run_experiment("bridge", function_pipeline, weights_wo_scalar, False)
    run_experiment("bridge", function_pipeline, weights, False)

    # %%
    print("Hist features stronger weights experiment!")
    run_experiment("helmet", function_pipeline, weights, False)
    run_experiment("helmet", function_pipeline, weights_w_strong_hist, False)
    print("================================================")
    run_experiment("ant", function_pipeline, weights, False)
    run_experiment("ant", function_pipeline, weights_w_strong_hist, False)
    print("================================================")
    run_experiment("bridge", function_pipeline, weights, False)
    run_experiment("bridge", function_pipeline, weights_w_strong_hist, False)
# %%

# %%
