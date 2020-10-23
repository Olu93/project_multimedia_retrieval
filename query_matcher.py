import io
from collections import ChainMap
from collections import OrderedDict
from reader import DataSet

from jsonlines.jsonlines import Reader
from normalizer import Normalizer
from pathlib import Path
from re import IGNORECASE

import jsonlines
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean, cityblock, sqeuclidean
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from helper.config import FEATURE_DATA_FILE

# TODO: [x] Display histograms
# TODO: [x] Check Distance function
# TODO: [] Check normalisation as shapes have different values
# TODO: [] Histograms values weight should be divided by n_bins so for one distributional feature to weight as other
#          features (i.e. line 142 each val/20), but when this is the case results are not well enough.


class QueryMatcher(object):
    IGNORE_COLUMNS = ["timestamp", "name", "label"]

    def __init__(self, extracted_feature_file):
        self.path_to_features = Path(extracted_feature_file)
        assert self.path_to_features.exists(), f"Feature file does not exist in {self.path_to_features.absolute().as_posix()}"
        self.features_raw = [data for data in jsonlines.Reader(io.open(self.path_to_features))]
        self.features_flattened = [QueryMatcher.flatten_feature_dict(feature_set) for feature_set in self.features_raw]
        self.features_df = pd.DataFrame(self.features_flattened).set_index('name').drop(columns="timestamp").drop(columns="label")
        self.features_column_names = list(self.features_df.columns)
        self.features_list_of_list = [QueryMatcher.prepare_for_matching(feature_set) for feature_set in self.features_raw]

    @staticmethod
    def init_from_query_mesh_features(feature_dict):
        features_flattened = [QueryMatcher.flatten_feature_dict(feature_set) for feature_set in feature_dict]
        features_df = pd.DataFrame(features_flattened)
        features_df = features_df.set_index('name').drop(columns="timestamp")
        return features_df

    @staticmethod
    def perform_knn(dataset, query, k):
        neighbors = NearestNeighbors(n_neighbors=k).fit(query)
        return neighbors.kneighbors(dataset)

    @staticmethod
    def compute_pca(data_matrix, n_components=50):
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        pca = PCA(n_components=n_components)
        projected = pca.fit(data_matrix.T)
        return projected.components_.T

    @staticmethod
    def compute_tsne(data_matrix, n_components=2):
        # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        X_embedded = TSNE(n_components=n_components).fit_transform(data_matrix)
        return X_embedded

    def compare_features_with_database(self, feature_set, k=5, distance_function=None):
        distance_function = QueryMatcher.cosine_similarity_faf if not distance_function else distance_function
        # Make order consistent with matching features db and flatten its distributional values
        feature_dict_in_correct_order = self.prepare_single_feature_for_comparison(feature_set, list(feature_set.columns))
        # Make an array of the flattened list and reshape so to be [1,]
        feature_instance_vector = np.array(list(feature_dict_in_correct_order.values())).reshape(1, -1)
        # Get processed features from json file
        feature_database_matrix = self.features_df.values
        # Create a matrix of query feature plus all other in db
        full_mat = np.vstack((feature_instance_vector, feature_database_matrix))
        # Standardise (zscore)
        full_mat[:, :6] = (full_mat[:, :6] - np.mean(full_mat[:, :6], axis=0)) / np.std(full_mat[:, :6], axis=0)
        # Because the cosine implemented by hand return most similar closer to zero we need to reverse in that case
        if distance_function == QueryMatcher.perform_knn:
            # Reduce matrix using PCA to alleviate computational load of TSNE,
            # default components is 50 as suggested in TSNE doc page here https://bit.ly/3j8ltDq
            full_mat = self.compute_pca(full_mat)
            # Perform t-distributed Stochastic Neighbor Embedding and reduce to default (n_shapes, 2) for
            # visualization porpoises
            # full_mat = self.compute_tsne(full_mat)
            # Perform knn and store results
            distance_values, indices = self.perform_knn(full_mat[0, :].reshape(1, -1), full_mat[1:, :], k)
        else:
            # Get results from distance function providing first row of matrix (query) and all others to match it with
            result = distance_function(full_mat[0, :].reshape(1, -1), full_mat[1:, :])
            # Get indices and values, but indices are not of the filename in db (i.e. 'm + index' won't work)
            indices, distance_values = QueryMatcher.get_top_k(result, k)
        # Retrieve the actual name from indexing the raw dict of shapes
        selected_shapes = np.array(self.features_raw)[indices]
        # For each shape selected, append name and return it
        names = [s["name"] for s in selected_shapes[0]]
        return names, distance_values

    def match_with_db(self, feature_set, k=5, distance_functions=[], weights=None):
        feature_set_transformed = QueryMatcher.prepare_for_matching(feature_set=feature_set)
        assert len(feature_set_transformed) == len(distance_functions), f"Not enough OR too many distance functions supplied!"
        all_distances = np.array([QueryMatcher.mono_run_functions_pipeline(feature_set_transformed, mesh_in_db, distance_functions, weights) for mesh_in_db in self.features_list_of_list])
        position_in_rank = np.argsort(all_distances)
        indices_of_smallest_distances = [list(position_in_rank).index(k_val) for k_val in range(k)]
        names = [mesh_in_db["name"] for mesh_in_db in np.array(self.features_raw)[indices_of_smallest_distances]]
        labels = [mesh_in_db["label"] for mesh_in_db in np.array(self.features_raw)[indices_of_smallest_distances]]
        
        # This sorts the results
        sorted_results = sorted(zip(all_distances[indices_of_smallest_distances], names, labels))
        sorted_values, sorted_names, sorted_labels = tuple(zip(*sorted_results))
        print(sorted_labels)
        return sorted_names, sorted_values
        # return names, distance_values

    @staticmethod
    def mono_run_functions_pipeline(a_features, b_features, dist_funcs, weights=None):
        """
        Runs the pipeline of functions and computes a comined value
        :param a_features: First mesh feature set
        :param b_features: Second mesh feature set
        :param dist_funcs: List of distance functions
        :param weightings: weights for which each distance functions takes part
        :return: Returns score for the distance
        """
        weights = [1] * len(dist_funcs) if not weights else weights

        return sum([w * fn(a, b) for a, b, fn, w in zip(a_features, b_features, dist_funcs, weights)])

    @staticmethod
    def prepare_for_matching(feature_set):
        """
        Acts as preparation for the matching process, as different features will use different distance functions.
        
        Puts scalar values into a single list. 
        Every distributional feature will be a single list. 
        In the all lists are combined into list of lists. 
        """
        scalar_features = [np.array([v for k, v in feature_set.items() if type(v) not in [np.ndarray, list] and k not in QueryMatcher.IGNORE_COLUMNS])]
        distributional_features = [np.array(v) for v in feature_set.values() if type(v) in [np.ndarray, list]]
        return scalar_features + distributional_features

    @staticmethod
    def get_top_k(cosine_similarities, k=5):
        top_k_indices = cosine_similarities.argsort(axis=1)[:, :k]
        # if not flipped else cosine_similarities.argsort(
        # axis=1)[:, -k:]
        taken = np.take(cosine_similarities, top_k_indices, axis=1)
        row_range = list(range(cosine_similarities.shape[0]))
        return top_k_indices, taken[row_range, row_range, :]

    # @staticmethod
    # def cosine_similarity_faf(A, B):
    #     nominator = np.dot(A, B.T)
    #     norm_A = np.linalg.norm(A, axis=1)
    #     norm_B = np.linalg.norm(B, axis=1)
    #     denominator = np.reshape(norm_A, [-1, 1]) * np.reshape(norm_B, [1, -1])
    #     return np.flip(np.divide(nominator, denominator))

    @staticmethod
    def wasserstein_distance(A, B):
        result = [wasserstein_distance(A.reshape(-1, ), B[r, :].reshape(-1, )) for r in range(B.shape[0])]
        return np.array(result).reshape(1, -1)

    @staticmethod
    def cosine_distance(A, B):
        result = [cosine(A.reshape(1, -1), B[r, :].reshape(1, -1)) for r in range(B.shape[0])]
        return np.array(result).reshape(1, -1)

    @staticmethod
    def euclidean_distance(A, B):
        result = [euclidean(A.reshape(1, -1), B[r, :].reshape(1, -1)) for r in range(B.shape[0])]
        return np.array(result).reshape(1, -1)

    @staticmethod
    def sqeuclidean_distance(A, B):
        result = [sqeuclidean(A.reshape(1, -1), B[r, :].reshape(1, -1)) for r in range(B.shape[0])]
        return np.array(result).reshape(1, -1)

    @staticmethod
    def manhattan_distance(A, B):
        result = [cityblock(A.reshape(1, -1), B[r, :].reshape(1, -1)) for r in range(B.shape[0])]
        return np.array(result).reshape(1, -1)

    @staticmethod
    def flatten_feature_dict(feature_set):
        singletons = {key: value for key, value in feature_set.items() if type(value) not in [list, np.ndarray]}
        distributional = [{f"{key}_{idx}": val for idx, val in enumerate(dist)} for key, dist in feature_set.items() if type(dist) in [list, np.ndarray]]
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
    sampled_mesh = qm.features_flattened[0]
    close_meshes, computed_values = qm.compare_features_with_database(pd.DataFrame(sampled_mesh, index=[0]), 5, QueryMatcher.cosine_distance)
    assert sampled_mesh["name"] in close_meshes
    function_pipeline = [cosine] + ([wasserstein_distance] * (len(qm.features_list_of_list[0])-1))
    print(QueryMatcher.mono_run_functions_pipeline(qm.features_list_of_list[0], qm.features_list_of_list[1], function_pipeline))
    print(qm.match_with_db(qm.features_raw[0], 5, function_pipeline))
    print("Everything worked!")

    # data = DataSet._read('ant.off')

