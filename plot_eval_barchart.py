from helper.config import FEATURE_DATA_FILE
from evaluator import Evaluator
import pandas as pd

from scipy.spatial.distance import cosine, euclidean, cityblock, sqeuclidean
from scipy.stats import wasserstein_distance

if __name__ == "__main__":
    evaluator = Evaluator(FEATURE_DATA_FILE, label_coarse=False)
    count_hists = sum([1 for header_name in evaluator.features_df_raw.columns if "hist_" in header_name])
    count_skeletons = sum([1 for header_name in evaluator.features_df_raw.columns if "skeleton_" in header_name])
    class_metric_means = pd.DataFrame(evaluator.mesh_classes_count.keys().sort_values().to_list(), columns=['class'])
    overall_metrics = []

    weights_best_all_features = ([4.157895]) + ([197.3684] * count_hists) + ([2.368421] * count_skeletons)
    function_pipeline_cowaeu = [cosine] + ([wasserstein_distance] * count_hists) + ([euclidean] * count_skeletons)

    # k_all_db_results_cowaeu = evaluator.perform_matching_calculate_metrics(function_pipeline_cowaeu,
    #                                                                         weights_best_all_features,
    #                                                                        k_full_db_switch=True)
    # k_all_db_results_cowaeu.to_json("k_all_db_results_cowaeu_bw.json")
    k_all_db_results_cowaeu = pd.read_json("k_all_db_results_cowaeu_bw.json")

    k_all_db_results_cowaeu = evaluator.metric_F1(k_all_db_results_cowaeu, evaluator.mesh_classes_count, k=10,
                                                  use_class_k=False, weightbool=True, weight=0.25)
    cowaeu_classes_mean_F1, cowaeu_overall_F1 = evaluator.calc_weighted_metric(k_all_db_results_cowaeu, "F1score", "F1")
    class_metric_means["F1_cowaeu"] = cowaeu_classes_mean_F1
    overall_metrics.append(["F1_cowaeu", cowaeu_overall_F1])
    evaluator.plot_metric_class_aggregate(class_metric_means[["class", "F1_cowaeu"]], "F1_cowaeu", "F1Score")
