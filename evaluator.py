from tqdm.std import tqdm
from helper.mp_functions import compute_matchings
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
# np.seterr('raise')

class Evaluator:
    def __init__(self):
        self.reader = DataSet("")
        self.query_matcher = QueryMatcher(FEATURE_DATA_FILE)
        self.feature_db_flattened = self.query_matcher.features_flattened
        self.feature_db_raw = self.query_matcher.features_raw
        self.features_df_flat = pd.DataFrame(self.feature_db_flattened).set_index('name').drop(columns="timestamp")
        self.features_df_raw = pd.DataFrame(self.feature_db_raw).set_index('name').drop(columns="timestamp")
        self.mesh_classes_count = self.features_df_flat['label'].value_counts()
        # Exclude classes which have only one member
        # self.mesh_classes_count = self.mesh_classes_count[self.mesh_classes_count > 1]

    def perform_matching_calculate_mean_average_metrics(self, function_pipeline, weights):
        match_results = pd.DataFrame(columns=['id', 'class', 'k', 'matches', 'matches_class', 'precision', 'recall', 'F1score'])
        top_3_classes = self.mesh_classes_count[:3]
        for index, row in enumerate(self.feature_db_raw):
            if row['label'] in top_3_classes.keys():
                for k in np.linspace((top_3_classes.get(row['label']) - top_3_classes.get(row['label']) / 3),
                                     (top_3_classes.get(row['label']) + (top_3_classes.get(row['label'])) / 3), 3):
                    ids, distance_values = self.query_matcher.match_with_db(row, k=int(k), distance_functions=function_pipeline, weights=weights)
                    ids.remove(row['name'])
                    matches_class = [self.features_df_raw[self.features_df_raw.index == id]['label'][0] for id in ids]
                    # Calculate metrics
                    # Correct matched meshes
                    TP = int(matches_class.count(row['label']))
                    # Meshes of the class that were not matched
                    FN = (self.mesh_classes_count.get(row['label']) - 1) - TP
                    # Matched meshes that have wrong class
                    FP = int(len(matches_class)) - TP
                    # All meshes th at were correctly not matched
                    TN = (((int(len(self.features_df_raw)) - TP) - FN) - FP)
                    # proportion of returned class from all returned items
                    precision = TP / (TP + FP)
                    # proportion of returned class from all class members in database
                    recall = TP / (TP + FN)
                    if (precision + recall) == 0:
                        F1score = 0
                    else:
                        F1score = 2 * ((precision * recall) / (precision + recall))
                    match_results = match_results.append(
                        {
                            'id': index,
                            'class': row['label'],
                            'k': int(k),
                            'matches': ids,
                            'matches_class': matches_class,
                            'precision': precision,
                            'recall': recall,
                            'F1score': F1score
                        },
                        ignore_index=True)
                    print(index, " of ", len(self.feature_db_raw))
        group = match_results.groupby(['class'])['F1score'].mean()
        return pd.DataFrame({"class": group.keys(), "F1score": group.values})

    @staticmethod
    def mono_compute_match(params):
        row, k, query_matcher, function_pipeline, weights = params
        ids, distance_values, clabels = query_matcher.match_with_db(row, k=k, distance_functions=function_pipeline, weights=weights)
        name = row["name"]
        label = row["label"]
        return name, label, ids, distance_values, clabels

    def perform_matching_calculate_metrics(self, function_pipeline, weights, k_full_db_switch=False):
        container_results = []
        param_list = []
        for row in self.feature_db_raw:
            k = np.sum(self.mesh_classes_count.values) if k_full_db_switch else (int(self.mesh_classes_count.get(row['label'])))
            param_list.append((row, k, self.query_matcher, function_pipeline, weights))

        for name, class_label, ids, distance_values, clabels in tqdm(compute_matchings(self, param_list), total=len(param_list)):

            # Query each shape over whole database and return k (nr of meshes for respective class)
            # if mesh from class that has been removed, continue to next mesh
            # if class_label in self.mesh_classes_count.keys():
            #     pass
            # else:
            #     continue
            # Remove the queried mesh from results
            pos_where_this_one_is = ids.index(name)
            ids.pop(pos_where_this_one_is)
            distance_values.pop(pos_where_this_one_is)
            clabels.pop(pos_where_this_one_is)
            # matches_class = [self.features_df_raw[self.features_df_raw.index == id]['label'][0] for id in ids]
            # Calculate metrics
            # Correct matched meshes
            TP = int(clabels.count(class_label))
            # Meshes of the class that were not matched
            FN = (self.mesh_classes_count.get(class_label) - 1) - TP
            # Matched meshes that have wrong class
            FP = int(len(clabels)) - TP
            # All meshes th at were correctly not matched
            TN = (((int(len(self.features_df_raw)) - TP) - FN) - FP)
            
            # proportion of returned class from all returned items
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            # proportion of returned class from all class members in database
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            F1score = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0

            container_results.append({
                'name': name,
                'class': class_label,
                'matches': ids,
                'matches_class': clabels,
                'distances': distance_values,
                'precision': precision,
                'recall': recall,
                'F1score': F1score
            })
            # print(name, " of ", len(self.feature_db_raw))
        match_results = pd.DataFrame(container_results, columns=['name', 'class', 'matches', 'matches_class', 'distances', 'precision', 'recall', 'F1score'])

        return match_results

    def plot_metric_class_aggregate(self, F1_means_list, used_func_list):
        barw = 0.8
        bottom = None
        bar_colors = ['b', 'y', 'g', 'r']
        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot()
        ax.tick_params(axis='x', rotation=90)
        if isinstance(F1_means_list[0], list):
            for i in range(len(F1_means_list)):
                if i != 0:
                    bottom = np.sum(F1_means_list[:i], axis=0)
                ax.bar(np.arange(0, (len(self.mesh_classes_count.keys()))), F1_means_list[i], bottom=bottom, color=bar_colors[i], width=barw, align='center')
            plt.xticks(np.arange(0, (len(self.mesh_classes_count.keys()))), self.mesh_classes_count.keys())
        else:
            ax.bar(np.arange(0, (len(F1_means_list[0]))), F1_means_list[0]['F1score'].to_list(), bottom=bottom, color='b', width=barw, align='center')
            plt.xticks(np.arange(0, (len(F1_means_list[0]))), F1_means_list[0]['class'].to_list())
        fig.tight_layout()
        ax.autoscale(tight=True)
        plt.tick_params(axis='x', which='major', labelsize=10)
        plt.tight_layout()
        plt.legend(used_func_list)
        plt.show()

    def calc_weighted_metric(self, results, metric):
        sum_of_items = np.sum(self.mesh_classes_count.values)
        weighted_class_metric_means = [
            (classid, (np.mean(results[results['class'] == classid][metric]) * self.mesh_classes_count.get(classid)) / sum_of_items)
            for classid in self.mesh_classes_count.keys()
        ]
        pprint(weighted_class_metric_means)
        weighted_class_metric_means_df = pd.DataFrame(weighted_class_metric_means, columns=["class", "bmean"])

        return weighted_class_metric_means_df, np.sum(weighted_class_metric_means_df['bmean'])


if __name__ == "__main__":
    evaluator = Evaluator()
    results_list = []
    used_func_list = []
    F1_result_means_list = []
    overall_mean_list = []
    overall_weighted_mean_list = []
    n_distributionals = get_sizes_features()[1]
    weights = ([1]) + ([1] * n_distributionals)

    # function_pipeline = [cosine] + ([cityblock] * (n_distributionals + 6))
    # mean_avg_F1scores = evaluator.perform_matching_calculate_mean_average_metrics(function_pipeline, weights)
    # mean_avg_F1scores = pd.read_csv("MAN_3classes.csv")
    # evaluator.plot_metric_class_aggregate([mean_avg_F1scores], ["Scalars: cosine, Dists: manhatten"])

    # k_all_db_results = evaluator.perform_matching_calculate_metrics(function_pipeline, weights, k_full_db_switch=True)
    # print("\n")
    # print(k_all_db_results.shape)
    # print(k_all_db_results.head())
    # k_all_db_results.to_csv("k_all_db_results.csv")
    
    # k_all_db_results = pd.read_csv("k_all_db_results.csv")
    # class_dists = pd.DataFrame(columns=['qclass', 'mclass', 'mean_dist'])
    # massive_list = []
    # for row in (k_all_db_results.iterrows()):
    #     massive_list.extend(list(zip(([row[1].get('class')] * len(eval(row[1].get('matches_class')))), eval(row[1].get('matches_class')), eval(row[1].get('distances')))))
    # class_dists_df = pd.DataFrame(massive_list, columns=['qclass', 'mclass', 'dist'])
    # grouped_aggregates = class_dists_df.groupby(['mclass', 'qclass']).mean()

    # pv_table = grouped_aggregates.reset_index().pivot_table(values='dist', index=['mclass'], columns='qclass')

    # num, div = len(evaluator.mesh_classes_count), 4
    # ranges = ([num // div + (1 if x < num % div else 0) for x in range(div)])
    # row_range = [0, ranges[0]]
    # col_range = [0, ranges[0]]
    # pop = ranges.pop(0)
    # ranges.append(ranges[-1])
    # for i1, val1 in enumerate(ranges):
    #     for i2, val2 in enumerate(ranges):
    #         print(row_range, col_range)
    #         # SCHINKEN IS HIER
    #         plt.figure(figsize=(40, 40))
    #         plt.rcParams["font.weight"] = "bold"
    #         plt.rcParams["axes.labelweight"] = "bold"
    #         ax = sns.heatmap(pv_table.iloc[row_range[0]:row_range[1], col_range[0]:col_range[1]], xticklabels=True, yticklabels=True, cbar=False)
    #         ax.xaxis.tick_top()
    #         ax.set_xlabel(" ")
    #         ax.set_ylabel(" ")
    #         ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=41, ha="left")
    #         ax.set_yticklabels(ax.get_yticklabels(), fontsize=41, ha="right")
    #         plt.savefig((str(i1) + str(i2) + ".png"))
    #         # SCHINKEEN
    #         col_range[0] = col_range[0] + val2
    #         col_range[1] = col_range[1] + val2
    #     col_range = [0, pop]
    #     row_range[0] = row_range[0] + val1
    #     row_range[1] = row_range[1] + val1

    # arr = np.array([])
    # for i, classid in enumerate(evaluator.mesh_classes_count.keys()):
    #     if i == 0:
    #         tmp_arr = np.array(grouped_aggregates.reset_index()[grouped_aggregates.reset_index()['mclass'] == classid]['dist'])
    #     else:
    #         tmp_arr = np.vstack((tmp_arr, np.array(grouped_aggregates.reset_index()[grouped_aggregates.reset_index()['mclass'] == classid]['dist'])))

    #     classes = eval(row[1].get("matches_class"))
    #     distances = eval(row[1].get("distances"))
    #     tmp_df = pd.DataFrame({"class": classes, "distances": distances})
    #     dists_grp_by_classes = tmp_df.groupby("class").mean()
    #     for i in range(len(dists_grp_by_classes)):
    #         class_dists = class_dists.append({'qclass': row[1].get('class'), 'mclass': dists_grp_by_classes.index.to_list()[i], 'mean_dist': dists_grp_by_classes['distances'].values.tolist()[i]}, ignore_index=True)
    # pass

    # BLOCK1 ------------
    # cosine & manhatten with equal weights 1
    function_pipeline = [cosine] + ([cityblock] * (n_distributionals + 6))
    cosine_manhatten_results = evaluator.perform_matching_calculate_metrics(function_pipeline, weights)
    # cosine_manhatten_results = pd.read_csv("cmr_results_df.csv")
    results_list.append(cosine_manhatten_results)
    used_func_list.append("Scalars: cosine, Dists: manhatten")
    cmr_F1_means = [np.mean(cosine_manhatten_results[cosine_manhatten_results['class'] == classid]['F1score']) for classid in evaluator.mesh_classes_count.keys()]
    F1_result_means_list.append(cmr_F1_means)
    cmr_overall_mean = np.mean(cmr_F1_means)
    overall_mean_list.append(cmr_overall_mean)
    cmr_weighted_means, cmr_overall_mean = evaluator.calc_weighted_metric(cosine_manhatten_results, "F1score")
    overall_weighted_mean_list.append(cmr_overall_mean)
    # # BLOCK1 ------------

    # # BLOCK2 ------------
    # # cosine & wasserstein with equal weights 1
    # # function_pipeline = [cosine] + ([wasserstein_distance] * n_distributionals)
    # # cosine_wasserstein_results = evaluator.perform_matching_calculate_metrics(function_pipeline, weights)
    # cosine_wasserstein_results = pd.read_csv("cwr_results_df.csv")
    # results_list.append(cosine_wasserstein_results)
    # used_func_list.append("Scalars: cosine, Dists: EMD")
    # cwr_F1_means = [np.mean(cosine_wasserstein_results[cosine_wasserstein_results['class'] == classid]['F1score']) for classid in evaluator.mesh_classes_count.keys()]
    # F1_result_means_list.append(cwr_F1_means)
    # cwr_overall_mean = np.mean(cwr_F1_means)
    # overall_mean_list.append(cwr_overall_mean)
    # cwr_weighted_means, cwr_overall_mean = evaluator.calc_weighted_metric(cosine_wasserstein_results, "F1score")
    # overall_weighted_mean_list.append(cwr_overall_mean)
    # # BLOCK2 ------------

    # # BLOCK3 ------------
    # # cosine & cosine with equal weights 1
    # # function_pipeline = [cosine] + ([cosine] * n_distributionals)
    # # cosine_cosine_results = evaluator.perform_matching_calculate_metrics(function_pipeline, weights)
    # cosine_cosine_results = pd.read_csv("ccr_results_df.csv")
    # results_list.append(cosine_cosine_results)
    # used_func_list.append("Scalars: cosine, Dists: cosine")
    # ccr_F1_means = [np.mean(cosine_cosine_results[cosine_cosine_results['class'] == classid]['F1score']) for classid in evaluator.mesh_classes_count.keys()]
    # F1_result_means_list.append(ccr_F1_means)
    # ccr_overall_mean = np.mean(ccr_F1_means)
    # overall_mean_list.append(ccr_overall_mean)
    # ccr_weighted_means, ccr_overall_mean = evaluator.calc_weighted_metric(cosine_cosine_results, "F1score")
    # overall_weighted_mean_list.append(ccr_overall_mean)
    # # BLOCK3 ------------

    evaluator.plot_metric_class_aggregate(F1_result_means_list, used_func_list)
    for i in range(len(overall_weighted_mean_list)):
        print("Weighted Overall Mean F1 Score - ", used_func_list[i], ": ", overall_weighted_mean_list[i])
        print("UN-Weighted Overall Mean F1 Score - ", used_func_list[i], ": ", overall_mean_list[i], '\n')
