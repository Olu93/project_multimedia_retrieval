import io
from collections import ChainMap
from collections import OrderedDict
from pathlib import Path
from scipy.stats import wasserstein_distance
import jsonlines
import numpy as np
import pandas as pd

from helper.config import FEATURE_DATA_FILE


# TODO: [] Display histograms
# TODO: [] Check Distance function

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
        # Make order consistent with matching features db and flatten its distributional values
        feature_dict_in_correct_order = self.prepare_single_feature_for_comparison(feature_set,
                                                                                   list(feature_set.columns))
        # Make an array of the flattened list and reshape so to be [1,]
        feature_instance_vector = np.array(list(feature_dict_in_correct_order.values())).reshape(1, -1)
        # Get processed features from json file
        feature_database_matrix = self.features_df.values
        # Create a matrix of query feature plus all other in db
        full_mat = np.vstack((feature_instance_vector, feature_database_matrix))
        # Replace NaN's with 0 and inf with 1, can be done better
        full_mat[np.isnan(full_mat)], full_mat[np.isinf(full_mat)] = 0, 1
        # Standardise (zscore)
        full_mat[:, :6] = (full_mat[:, :6] - np.mean(full_mat[:, :6], axis=0)) / np.std(full_mat[:, :6], axis=0)
        # Get results from distance function
        result = distance_function(full_mat[0, :].reshape(1, -1), full_mat[1:, :])
        # Get indices and values, but indices are not of the filename in db (i.e. 'm + index' won't work)
        indices, cosine_values = QueryMatcher.get_top_k(result, k)
        # Retrieve the actual name from indexing the raw dict of shapes
        selected_shapes = np.array(self.features_raw)[indices]
        # For each shape selected, append name and return it
        names = [s["name"] for s in selected_shapes[0]]
        return names, cosine_values

    @staticmethod
    def get_top_k(cosine_similarities, k=5):
        cosine_similarities = np.nan_to_num(cosine_similarities)
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
    def wasserstein_distance(A, B):
        pass

    @staticmethod
    def flatten_feature_dict(feature_set):
        singletons = {key: value for key, value in feature_set.items() if type(value) not in [list, np.ndarray]}
        distributional = [{f"{key}_{idx}": val for idx, val in enumerate(dist)} for key, dist in feature_set.items() if
                          type(dist) in [list, np.ndarray]]

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

    @staticmethod
    def normalise_hist(distribution):
        return np.sum(distribution)


if __name__ == "__main__":
    qm = QueryMatcher(FEATURE_DATA_FILE)
    print(len(qm.features_raw))
    print(qm.compare_features_with_database(qm.features_raw[0]))
