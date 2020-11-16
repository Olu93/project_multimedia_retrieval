from tqdm.std import tqdm
from helper.mp_functions import compute_matchings, compute_matchings_old
import numpy as np
import pandas as pd
from reader import DataSet
from query_matcher import QueryMatcher
from helper.config import FEATURE_DATA_FILE, DATA_PATH_NORMED, DATA_PATH_PSB
from helper.misc import get_sizes_features
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean, cityblock, sqeuclidean
from scipy.stats import wasserstein_distance
from feature_extractor import FeatureExtractor
import seaborn as sns
from pprint import pprint
import jsonlines
# np.seterr('raise')

class Evaluator:
    def __init__(self, feature_data_file=FEATURE_DATA_FILE, label_coarse=False):
        self.label_coarse = label_coarse
        self.reader = DataSet("")
        self.query_matcher = QueryMatcher(feature_data_file, label_coarse=label_coarse)
        self.feature_db_flattened = self.query_matcher.features_flattened
        self.feature_db_raw = self.query_matcher.features_raw
        self.features_df_flat = pd.DataFrame(self.feature_db_flattened).set_index('name')
        self.features_df_raw = pd.DataFrame(self.feature_db_raw).set_index('name')
        self.mesh_classes_count = self.features_df_flat['label'].value_counts()
        if self.label_coarse:
            self.mesh_classes_count = self.features_df_flat['label_coarse'].value_counts().rename("label")
            # self.mesh_classes_count.
        # print(self.mesh_classes_count_coarse.name)
        # Exclude classes which have only one member
        # self.mesh_classes_count = self.mesh_classes_count[self.mesh_classes_count > 1]

    def perform_matching_on_subset(self, subset, function_pipeline, weights, k_full_db_switch=False):
        param_list = []
        for row in subset:
            row["label"] = row['label_coarse'] if self.label_coarse else row["label"]
            k = np.sum(self.mesh_classes_count.values) if k_full_db_switch else (int(self.mesh_classes_count.get(row['label' if self.label_coarse else 'label_coarse'])))
            param_list.append((row, k, self.query_matcher, function_pipeline, weights))
        match_results = pd.DataFrame(list(compute_matchings(self, param_list)), columns=['name', 'class', 'matches', 'distances', 'matches_class'])
        return match_results

    def perform_matching_on_subset_slow(self, subset, function_pipeline, weights, k_full_db_switch=False):
        param_list = []
        for row in subset:
            row["label"] = row['label_coarse'] if self.label_coarse else row["label"]
            k = np.sum(self.mesh_classes_count.values) if k_full_db_switch else (int(self.mesh_classes_count.get(row['label'])))
            param_list.append((row, k, self.query_matcher, function_pipeline, weights))
        match_results = pd.DataFrame(list([self.mono_compute_match(param) for param in param_list]), columns=['name', 'class', 'matches', 'distances', 'matches_class'])
        return match_results

    @staticmethod
    def mono_compute_match(params):
        row, k, query_matcher, function_pipeline, weights = params
        ids, distance_values, clabels = query_matcher.match_with_db(row, k=k, distance_functions=function_pipeline, weights=weights)
        name = row["name"]
        label = row["label"]
        return name, label, ids, distance_values, clabels

    def perform_matching_calculate_metrics(self, function_pipeline, weights, k_full_db_switch=False):
        param_list = []
        for row in self.feature_db_raw:
            k = np.sum(self.mesh_classes_count.values) if k_full_db_switch else (int(self.mesh_classes_count.get(row['label'])))
            param_list.append((row, k, self.query_matcher, function_pipeline, weights))
        match_results = pd.DataFrame(list(compute_matchings(self, param_list)), columns=['name', 'class', 'matches', 'distances','matches_class'])
        return match_results

    # Uses flattened version with same dist func for all features
    @staticmethod
    def mono_compute_match_old(params):
        row, k, query_matcher, sfunc, hfunc, weights = params
        ids, distance_values, clabels = query_matcher.compare_features_with_database(pd.DataFrame(row, index=[0]), weights, k=k, scalar_dist_func=sfunc, hist_dist_func=hfunc)
        name = row["name"]
        label = row["label"]
        return name, label, ids, distance_values, clabels

    def perform_matching_calculate_metrics_old(self, sfunc, hfunc, weights, k_full_db_switch=False):
        param_list = []
        for row in self.feature_db_flattened:
            k = np.sum(self.mesh_classes_count.values) if k_full_db_switch else (int(self.mesh_classes_count.get(row['label'])))
            param_list.append((row, k, self.query_matcher, sfunc, hfunc, weights))
        match_results = pd.DataFrame(list(compute_matchings_old(self, param_list)), columns=['name', 'class', 'matches', 'distances','matches_class'])
        return match_results


    @staticmethod
    def metric_F1(k_all_db_result, all_class_counts_list, k=10, use_class_k=True, weightbool=False, weight=1, silent=False):
        def prepare_params(name, class_label, ids, distance_values, clabels):
            use_k = all_class_counts_list.get(class_label) if use_class_k else k
            return Evaluator.confusion_matrix_vals(name, class_label, ids[:use_k], distance_values[:use_k], clabels[:use_k], all_class_counts_list)
        data_generator = tqdm(k_all_db_result.to_numpy()) if not silent else k_all_db_result.to_numpy()
        cm_vals_and_label = [(prepare_params(*params), params[1]) for params in data_generator]
        cm_vals = np.array([results for results, _ in cm_vals_and_label])
        TP, FP, TN, FN = cm_vals[:, 0], cm_vals[:, 1], cm_vals[:, 2], cm_vals[:, 3]

        # precision = proportion of returned class from all returned items
        # recall = proportion of returned class from all class members in database
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.nan_to_num(TP / (TP + FP))
            recall = np.nan_to_num(TP / (TP + FN))
            if not weightbool:
                F1scores = np.nan_to_num(2 * ((precision * recall) / (precision + recall)))
                k_all_db_result['F1score'] = F1scores
                k_all_db_result['F1precision'] = precision
                k_all_db_result['F1recall'] = recall
            else:
                F1scores = np.nan_to_num((1 + np.square(weight)) * ((precision * recall) / ((np.square(weight) * precision) + recall)))
                k_all_db_result['F1score'] = F1scores
                k_all_db_result['F1precision'] = precision
                k_all_db_result['F1recall'] = recall



        # F1_list.append({'F1score': F1score})
        k_all_db_result['F1score'] = F1scores
        k_all_db_result['F1precision'] = precision
        k_all_db_result['F1recall'] = recall
        return k_all_db_result

    @staticmethod
    def metric_Ktier(k_all_db_result):
        pass

    @staticmethod
    def confusion_matrix_vals(name, class_label, ids, distance_values, clabels, all_class_counts_list):
        # qid = single_query_result['id']
        # qclass = single_query_result['class']
        # matches_ids = single_query_result['matches']
        # matches_class = single_query_result['matches_class']
        # Remove the queried mesh from results
        pos_where_this_one_is = ids.index(name)
        ids.pop(pos_where_this_one_is)
        clabels.pop(pos_where_this_one_is)
        # Calc confusion matrix values
        TP = int(clabels.count(class_label))
        # Meshes of the class that were not matched
        # print(class_label)
        if False:
            print(f"{class_label} - {name} - {TP} hits!")
        FN = (all_class_counts_list.get(class_label) - 1) - TP
        # Matched meshes that have wrong class
        FP = int(len(clabels)) - TP
        # All meshes th at were correctly not matched
        TN = (((np.sum(all_class_counts_list.values) - TP) - FN) - FP)

        return TP, FP, TN, FN

    def plot_metric_class_aggregate(self, classes_and_metrics, metric_col_name, metric_plot_label):
        fig = plt.figure(figsize=(20, 5))
        pal = sns.color_palette("Greens_d", len(classes_and_metrics))
        rank = classes_and_metrics.sort_values(by=metric_col_name, ascending=True)[metric_col_name].argsort().argsort()
        ax = sns.barplot(data=classes_and_metrics.sort_values(by=metric_col_name, ascending=True), x='class', y=metric_col_name, palette=np.array(pal)[rank])
        ax.tick_params(axis='x', rotation=90)
        fig.tight_layout()
        ax.autoscale(tight=True)
        plt.tick_params(axis='x', which='major', labelsize=10)
        plt.tight_layout()
        plt.ylabel(metric_plot_label)
        plt.savefig((metric_col_name + "_barc.png"))
        plt.show()

    @staticmethod
    def calc_weighted_metric(dataframe_with_metrics, metric, descr=None):
        mean_per_class = dataframe_with_metrics[['class', metric]].groupby('class').mean()
        overall_metric = np.mean(dataframe_with_metrics[metric].values)
        if descr:
            print(descr, overall_metric)
        return mean_per_class.sort_values(by='class', ascending=True)[metric].values, overall_metric

    @staticmethod
    def generate_class_dists_heatmap(k_all_results, slice=False, descr="heatmap1", div_classes_len=4):
        massive_list = []
        for row in (k_all_results.iterrows()):
            massive_list.extend(list(zip(([row[1].get('class')] * len(row[1].get('matches_class'))), row[1].get('matches_class'), row[1].get('distances'))))
        class_dists_df = pd.DataFrame(massive_list, columns=['qclass', 'mclass', 'dist'])
        grouped_aggregates = class_dists_df.groupby(['mclass', 'qclass']).mean()
        pv_table = grouped_aggregates.reset_index().pivot_table(values='dist', index=['mclass'], columns='qclass')
        if not slice:
            # SCHINKEN IS HIER
            plt.figure(figsize=(20, 15))
            sns.set(font_scale=2)
            # cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
            # plt.rcParams["font.weight"] = "bold"
            # plt.rcParams["axes.labelweight"] = "bold"
            ax = sns.heatmap(pv_table,  xticklabels=[""], yticklabels=[""], vmax=100, vmin=0)
            cbar = ax.collections[0].colorbar
            cbar.set_ticks([0, 20, 75, 100])
            cbar.set_ticklabels(['short distance', ' ', ' ', 'large distance'])
            ax.set_xlabel("classes")
            ax.set_ylabel("classes")
            plt.savefig((descr + ".png"))
            # SCHINKEEN
        else:
            num = len(evaluator.mesh_classes_count)
            ranges = ([num // div_classes_len + (1 if x < num % div_classes_len else 0) for x in range(div_classes_len)])
            row_range = [0, ranges[0]]
            col_range = [0, ranges[0]]
            pop = ranges.pop(0)
            ranges.append(ranges[-1])
            for i1, val1 in enumerate(ranges):
                for i2, val2 in enumerate(ranges):
                    print(row_range, col_range)
                    # SCHINKEN IS HIER
                    plt.figure(figsize=(40, 40))
                    plt.rcParams["font.weight"] = "bold"
                    plt.rcParams["axes.labelweight"] = "bold"
                    ax = sns.heatmap(pv_table.iloc[row_range[0]:row_range[1], col_range[0]:col_range[1]], xticklabels=True, yticklabels=True, cbar=False)
                    ax.xaxis.tick_top()
                    ax.set_xlabel(" ")
                    ax.set_ylabel(" ")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=41, ha="left")
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=41, ha="right")
                    plt.savefig((descr + str(i1) + str(i2) + ".png"))
                    # SCHINKEEN
                    col_range[0] = col_range[0] + val2
                    col_range[1] = col_range[1] + val2
                col_range = [0, pop]
                row_range[0] = row_range[0] + val1
                row_range[1] = row_range[1] + val1


if __name__ == "__main__":
    evaluator = Evaluator()
    class_metric_means = pd.DataFrame(evaluator.mesh_classes_count.keys().sort_values().to_list(), columns=['class'])
    overall_metrics = []
    count_hists = sum([1 for header_name in evaluator.features_df_raw.columns if "hist_" in header_name])
    count_skeletons = sum([1 for header_name in evaluator.features_df_raw.columns if "skeleton_" in header_name])
    n_distributionals = len(FeatureExtractor.get_pipeline_functions()[1])

    weights_with_skele_3_100_1 = ([3]) + ([100] * n_distributionals) + ([1] * count_skeletons)
    weights_no_skele = ([1]) + ([1] * count_hists) + ([0] * count_skeletons)


    feature_selection = (1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0)
    weights_best_all_features = ([4.157895]) + ([197.3684] * count_hists) + ([2.368421] * count_skeletons)
    weights_best_selected_features = np.multiply(weights_best_all_features, feature_selection).tolist()

    # function_pipeline_cociwa = [cosine] + ([cityblock] * n_distributionals) + ([wasserstein_distance] * count_skeletons)
    # function_pipeline_cowawa = [cosine] + ([wasserstein_distance] * n_distributionals) + ([wasserstein_distance] * count_skeletons)
    # function_pipeline_cocowa = [cosine] + ([cosine] * n_distributionals) + ([wasserstein_distance] * count_skeletons)

    function_pipeline_cociwa = [cosine] + ([cityblock] * count_hists) + ([wasserstein_distance] * count_skeletons)
    function_pipeline_cowawa = [cosine] + ([wasserstein_distance] * count_hists) + ([wasserstein_distance] * count_skeletons)
    function_pipeline_cowaeu = [cosine] + ([wasserstein_distance] * count_hists) + ([euclidean] * count_skeletons)
    function_pipeline_cocowa = [cosine] + ([cosine] * count_hists) + ([wasserstein_distance] * count_skeletons)
    function_pipeline_cocici = [cosine] + ([cityblock] * count_hists) + ([cityblock] * count_skeletons)


    # # Cosine + Mannhaten
    # print("COCI\n")
    # # k_all_db_results_coci = evaluator.perform_matching_calculate_metrics(function_pipeline_cociwa, weights_no_skele, k_full_db_switch=True)
    # # k_all_db_results_coci.to_json("k_all_db_results_coci.json")
    # k_all_db_results_coci = pd.read_json("k_all_db_results_coci.json")
    # k_all_db_results_coci = evaluator.metric_F1(k_all_db_results_coci, evaluator.mesh_classes_count, k=10, use_class_k=True)
    # coci_classes_mean_F1, coci_overall_F1 = evaluator.calc_weighted_metric(k_all_db_results_coci, "F1score", "F1")
    # class_metric_means["F1_coci"] = coci_classes_mean_F1
    # overall_metrics.append(["F1_coci", coci_overall_F1])
    # evaluator.plot_metric_class_aggregate(class_metric_means[["class", "F1_coci"]], "F1_coci", "F1Score")

    # # Cosine + Mannhaten + EMD
    # print("COCIWA\n")
    # # k_all_db_results_cociwa = evaluator.perform_matching_calculate_metrics(function_pipeline_cociwa, weights_with_skele_3_100_1,
    # #                                                                        k_full_db_switch=True)
    # # k_all_db_results_cociwa.to_json("k_all_db_results_cociwa.json")
    # k_all_db_results_cociwa = pd.read_json("k_all_db_results_cociwa.json")
    # k_all_db_results_cociwa = evaluator.metric_F1(k_all_db_results_cociwa, evaluator.mesh_classes_count, k=10, use_class_k=True)
    # cociwa_classes_mean_F1, cociwa_overall_F1 = evaluator.calc_weighted_metric(k_all_db_results_cociwa, "F1score", "F1")
    # class_metric_means["F1_cociwa"] = cociwa_classes_mean_F1
    # overall_metrics.append(["F1_cociwa", cociwa_overall_F1])
    # evaluator.plot_metric_class_aggregate(class_metric_means[["class", "F1_cociwa"]], "F1_cociwa", "F1Score")


    # # Cosine + EMD
    # print("COWA\n")
    # # k_all_db_results_cowa = evaluator.perform_matching_calculate_metrics(function_pipeline_cowawa, weights_no_skele, k_full_db_switch=True)
    # # k_all_db_results_cowa.to_json("k_all_db_results_cowa.json")
    # k_all_db_results_cowa = pd.read_json("k_all_db_results_cowa.json")
    # k_all_db_results_cowa = evaluator.metric_F1(k_all_db_results_cowa, evaluator.mesh_classes_count, k=10, use_class_k=True)
    # cowa_classes_mean_F1, cowa_overall_F1 = evaluator.calc_weighted_metric(k_all_db_results_cowa, "F1score", "F1")
    # class_metric_means["F1_cowa"] = cowa_classes_mean_F1
    # overall_metrics.append(["F1_cowa", cowa_overall_F1])
    # evaluator.plot_metric_class_aggregate(class_metric_means[["class", "F1_cowa"]], "F1_cowa", "F1Score")
    #
    #
    #
    # # Cosine + EMD + EMD
    # print("COWAWA\n")
    # # k_all_db_results_cowawa = evaluator.perform_matching_calculate_metrics(function_pipeline_cowawa,
    # #                                                                           weights_with_skele_3_100_1, k_full_db_switch=True)
    # # k_all_db_results_cowawa.to_json("k_all_db_results_cowawa.json")
    # k_all_db_results_cowawa = pd.read_json("k_all_db_results_cowawa.json")
    # k_all_db_results_cowawa = evaluator.metric_F1(k_all_db_results_cowawa, evaluator.mesh_classes_count, k=10, use_class_k=True)
    # cowawa_classes_mean_F1, cowawa_overall_F1 = evaluator.calc_weighted_metric(k_all_db_results_cowawa, "F1score", "F1")
    # class_metric_means["F1_cowawa"] = cowawa_classes_mean_F1
    # overall_metrics.append(["F1_cowawa", cowawa_overall_F1])
    # evaluator.plot_metric_class_aggregate(class_metric_means[["class", "F1_cowawa"]], "F1_cowawa", "F1Score")

    # Cosine + EMD + Euclidean
    print("COWAEU_1\n")
    # k_all_db_results_cowaeu = evaluator.perform_matching_calculate_metrics(function_pipeline_cowaeu,
    #                                                                        weights_best_all_features,
    #                                                                        k_full_db_switch=True)
    # k_all_db_results_cowaeu.to_json("k_all_db_results_cowaeu_bw.json")
    k_all_db_results_cowaeu = pd.read_json("k_all_db_results_cowaeu_bw.json")

    evaluator.generate_class_dists_heatmap(k_all_db_results_cowaeu, True, "slices", 4)

    # print(evaluator.mesh_classes_count.values)
    # k_all_db_results_cowaeu = evaluator.metric_F1(k_all_db_results_cowaeu, evaluator.mesh_classes_count, k=10,
    #                                               use_class_k=False, weightbool=True, weight=0.25)
    # cowaeu_classes_mean_F1, cowaeu_overall_F1 = evaluator.calc_weighted_metric(k_all_db_results_cowaeu, "F1score", "F1")
    # class_metric_means["F1_cowaeu"] = cowaeu_classes_mean_F1
    # overall_metrics.append(["F1_cowaeu", cowaeu_overall_F1])
    # evaluator.plot_metric_class_aggregate(class_metric_means[["class", "F1_cowaeu"]], "F1_cowaeu", "F1Score")

    # print("COWAEU_2\n")
    # k_all_db_results_cowaeu = evaluator.perform_matching_calculate_metrics(function_pipeline_cowaeu,
    #                                                                        weights_best_selected_features,
    #                                                                        k_full_db_switch=True)
    # k_all_db_results_cowaeu.to_json("k_all_db_results_cowaeu_bwsf.json")
    # k_all_db_results_cowaeu = pd.read_json("k_all_db_results_cowaeu.json")
    # k_all_db_results_cowaeu = evaluator.metric_F1(k_all_db_results_cowaeu, evaluator.mesh_classes_count, k=10,
    #                                               use_class_k=True)
    # cowaeu_classes_mean_F1, cowaeu_overall_F1 = evaluator.calc_weighted_metric(k_all_db_results_cowaeu, "F1score", "F1")
    # class_metric_means["F1_cowaeu"] = cowaeu_classes_mean_F1
    # overall_metrics.append(["F1_cowaeu", cowaeu_overall_F1])
    # evaluator.plot_metric_class_aggregate(class_metric_means[["class", "F1_cowaeu"]], "F1_cowaeu", "F1Score")

    # Cosine + Mannhaten + Manhatten
    # print("COCICI\n")
    # k_all_db_results_cocici = evaluator.perform_matching_calculate_metrics(function_pipeline_cocici,
    #                                                                        weights_with_skele_3_100_1,
    #                                                                        k_full_db_switch=True)
    # k_all_db_results_cocici.to_json("k_all_db_results_cocici.json")
    # k_all_db_results_cocici = pd.read_json("k_all_db_results_cocici.json")
    # k_all_db_results_cocici = evaluator.metric_F1(k_all_db_results_cocici, evaluator.mesh_classes_count, k=10,
    #                                               use_class_k=True)
    # cocici_classes_mean_F1, cocici_overall_F1 = evaluator.calc_weighted_metric(k_all_db_results_cocici, "F1score", "F1")
    # class_metric_means["F1_cocici"] = cocici_classes_mean_F1
    # overall_metrics.append(["F1_cocici", cocici_overall_F1])
    # evaluator.plot_metric_class_aggregate(class_metric_means[["class", "F1_cocici"]], "F1_cocici", "F1Score")


    # FLATTENED VERSION Cosine + Manhatten
    # k_all_db_results_coci = evaluator.perform_matching_calculate_metrics_old(cosine, cityblock, [1, 1], k_full_db_switch= True)
    # k_all_db_results_coci.to_json("k_all_db_results_coci.json")
    # k_all_db_results_coci = pd.read_json("k_all_db_results_coci.json")
    # k_all_db_results_coci = evaluator.metric_F1(k_all_db_results_coci, evaluator.mesh_classes_count, k=10, use_class_k= True)


    # evaluator.generate_class_dists_heatmap(k_all_db_results_cociwa, True, "lowlvl_classes", 4)




