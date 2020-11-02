# %%
from collections import Counter
from scipy.spatial.distance import cityblock, cosine
from scipy.stats.stats import wasserstein_distance
from feature_extractor import FeatureExtractor
from helper.config import FEATURE_DATA_FILE
from evaluator import Evaluator
import pandas as pd

if __name__ == "__main__":
    evaluator = Evaluator(FEATURE_DATA_FILE, label_coarse=False)
    mapping = {item["label"]: item["label_coarse"] for item in evaluator.query_matcher.features_raw_init}

    class_metric_means = pd.DataFrame(evaluator.mesh_classes_count.keys().sort_values().to_list(), columns=['class'])
    overall_metrics = []
    count_hists = sum([1 for header_name in evaluator.features_df_raw.columns if "hist_" in header_name])
    count_skeletons = sum([1 for header_name in evaluator.features_df_raw.columns if "skeleton_" in header_name])
    n_distributionals = len(FeatureExtractor.get_pipeline_functions()[1])

    weights = ([1]) + ([1] * n_distributionals) + ([1] * count_skeletons)

    function_pipeline_cociwa = [cosine] + ([wasserstein_distance] * n_distributionals) + ([cityblock] * count_skeletons)
    function_pipeline_cowawa = [cosine] + ([wasserstein_distance] * n_distributionals) + ([wasserstein_distance] * count_skeletons)
    function_pipeline_cocowa = [cosine] + ([cosine] * n_distributionals) + ([wasserstein_distance] * count_skeletons)

    all_data = evaluator.query_matcher.features_raw
    # print(evaluator.mesh_classes_count.keys())
    query_class = "helmet"
    my_subset = [item for item in all_data if item["label"] == (mapping[query_class] if evaluator.label_coarse else query_class)]
    k_all_db_results_cociwa = evaluator.perform_matching_on_subset_slow(my_subset, function_pipeline_cociwa, weights, k_full_db_switch=True)
    k_all_db_results_cociwa = evaluator.metric_F1(k_all_db_results_cociwa, evaluator.mesh_classes_count, k=10, use_class_k=False)

    # %%
    k_all_db_results_cociwa[["matches_class"]] = [Counter(tuple(row[["matches_class"]][0])) for index, row in k_all_db_results_cociwa.iterrows()]
    results = k_all_db_results_cociwa[["name", "class", "matches_class"]]
    print(results.head(10))
    results.head(10)
    # %%
    print(evaluator.calc_weighted_metric(k_all_db_results_cociwa)[1])
    # %%
