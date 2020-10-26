import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from helper.config import FEATURE_DATA_FILE
from query_matcher import QueryMatcher
from reader import DataSet


class Evaluator:
    def __init__(self):
        self.reader = DataSet("")
        self.query_matcher = QueryMatcher(FEATURE_DATA_FILE)
        self.feature_db_flattened = self.query_matcher.features_flattened
        self.features_df = pd.DataFrame(self.feature_db_flattened).set_index('name').drop(columns="timestamp")
        self.mesh_classes_count = self.features_df['label'].value_counts()
        # Exclude classes which have only one member
        self.mesh_classes_count = self.mesh_classes_count[self.mesh_classes_count > 1]

    def perform_matching_calculate_metrics(self, query_function):
        match_results = pd.DataFrame(
            columns=['id', 'class', 'matches', 'matches_class', 'precision', 'recall', 'F1score'])
        for index, row in self.features_df.iterrows():
            # Query each shape over whole database and return k (nr of meshes for respective class)
            row = self.features_df[self.features_df.index == index]
            class_label = row['label'].values[0]
            # if mesh from class that has been removed, continue to next mesh
            if class_label in self.mesh_classes_count.keys():
                # k = amount of meshes of the class minus the one which is used to query
                class_k = (int(self.mesh_classes_count.get(class_label)) - 1)
            else:
                continue
            ids, distance_values = self.query_matcher.compare_features_with_database(row.drop(columns="label"),
                                                                                     k=class_k,
                                                                                     distance_function=query_function)
            # Remove the queried mesh from results
            ids.remove(index)
            matches_class = [self.features_df[self.features_df.index == id]['label'].values[0] for id in ids]

            # Calculate metrics
            # Correct matched meshes
            TP = int(matches_class.count(class_label))
            # Meshes of the class that were not matched
            FN = int(class_k) - TP
            # Matched meshes that have wrong class
            FP = int(len(matches_class)) - TP
            # All meshes that were correctly not matched
            TN = (((int(len(self.features_df)) - TP) - FN) - FP)

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            if (precision + recall) == 0:
                F1score = 0
            else:
                F1score = 2 * ((precision * recall) / (precision + recall))

            match_results = match_results.append(
                {'id': index, 'class': class_label, 'matches': ids, 'matches_class': matches_class,
                 'precision': precision, 'recall': recall, 'F1score': F1score}, ignore_index=True)

        return match_results

    def plot_metric_class_aggregate(self, results_df, str_distance_function):
        F1means = [np.mean(results_df[results_df['class'] == classid]['F1score']) for classid in
                   self.mesh_classes_count.keys()]
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot()
        ax.tick_params(axis='x', rotation=90)
        ax.bar(self.mesh_classes_count.keys(), F1means)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    evaluator = Evaluator()
    cosine_results = evaluator.perform_matching_calculate_metrics(cosine)
    evaluator.plot_metric_class_aggregate(cosine_results, "cosine")
