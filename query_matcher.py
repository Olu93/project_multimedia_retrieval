from helper.config import DEBUG, DATA_PATH_NORMED_SUBSET, FEATURE_DATA_FILE, DATA_PATH_NORMED, DEBUG, DATA_PATH_NORMED_SUBSET, CLASS_FILE
import jsonlines
import io
from os import path
import pyvista as pv
from pathlib import Path
from collections import ChainMap
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler


class QueryMatcher(object):
    IGNORE_COLUMNS = ["timestamp", "name", "label"]

    def __init__(self, extracted_feature_file):
        self.path_to_features = Path(extracted_feature_file)
        assert self.path_to_features.exists(), f"Feature file does not exist in {self.path_to_features.absolute().as_posix()}"
        self.features_raw = [data for data in jsonlines.Reader(io.open(self.path_to_features))]
        self.features_flattened = [QueryMatcher.flatten_feature_dict(feature_set) for feature_set in self.features_raw]
        self.features_df = pd.DataFrame(self.features_flattened)
        self.features_df = self.features_df.set_index('name').drop(columns="timestamp")
        self.features_column_names = list(self.features_df.columns)

    @staticmethod
    def init_from_query_mesh_features(feature_dict):
        features_flattened = [QueryMatcher.flatten_feature_dict(feature_set) for feature_set in feature_dict]
        features_df = pd.DataFrame(features_flattened)
        features_df = features_df.set_index('name').drop(columns="timestamp")
        features_column_names = list(features_df.columns)
        return features_df

    def compare_features_with_database(self, feature_set, k=5, distance_function=None):
        distance_function = QueryMatcher.cosine_similarity_faf if not distance_function else distance_function
        # feature_dict_in_correct_order = self.prepare_single_feature_for_comparison(feature_set, self.features_column_names)
        feature_dict_in_correct_order = self.prepare_single_feature_for_comparison(feature_set, list(feature_set.columns))
        feature_instance_vector = np.array(list(feature_dict_in_correct_order.values())).reshape(1, -1)
        feature_database_matrix = self.features_df.values
        result = distance_function(feature_instance_vector, feature_database_matrix)
        indices, cosine_values = QueryMatcher.get_top_k(result, k)
        return indices, cosine_values


    @staticmethod
    def get_top_k(cosine_similarities, k=5):
        top_k_indices = cosine_similarities.argsort(axis=1)[:, -k:]
        taken = np.take(cosine_similarities, top_k_indices, axis=1)
        row_range = list(range(cosine_similarities.shape[0]))
        return top_k_indices, taken[row_range, row_range, :]

    @staticmethod
    def cosine_similarity_faf(A, B):
        nominator = np.dot(A, B.T)
        norm_A = np.linalg.norm(A, axis=1)
        norm_B = np.linalg.norm(B, axis=1)
        denominator = np.reshape(norm_A, [-1, 1]) * np.reshape(norm_B, [1, -1])
        return np.divide(nominator, denominator)

    @staticmethod
    def flatten_feature_dict(feature_set):
        singletons = {key: value for key, value in feature_set.items() if type(value) not in [list, np.ndarray]}
        distributional = [{f"{key}_{idx}": val for idx, val in enumerate(dist)} for key, dist in feature_set.items() if type(dist) in [list, np.ndarray]]
        # flattened_feature_set = dict((pair for d in distributional for pair in d.items()))
        flattened_feature_set = dict(ChainMap(*distributional, singletons))
        return flattened_feature_set

    @staticmethod
    def prepare_single_feature_for_comparison(feature_set, columns_in_order):
        dict_in_order = OrderedDict()
        features = QueryMatcher.flatten_feature_dict(feature_set)
        for col in columns_in_order:
            if col in QueryMatcher.IGNORE_COLUMNS:
                continue
            dict_in_order[col] = float(features.get(col, np.nan))
        return dict_in_order


if __name__ == "__main__":
    qm = QueryMatcher(FEATURE_DATA_FILE)
    print(len(qm.features_raw))
    # pprint()
    print(qm.compare_features_with_database(qm.features_raw[0]))