import io
import os.path
from collections import ChainMap
from collections import OrderedDict
from pathlib import Path

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial.distance import cosine, euclidean, cityblock, sqeuclidean
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from feature_extractor import FeatureExtractor
from helper.config import FEATURE_DATA_FILE
from helper.misc import rand_cmap
from normalizer import Normalizer
from reader import DataSet


# TODO: k-nearest takes three arguments so it just displays k=5, doesnt update with user chosen k

class QueryMatcher(object):
    IGNORE_COLUMNS = ["timestamp", "name", "label"]

    def __init__(self, extracted_feature_file):
        self.scaler = StandardScaler()
        self.path_to_features = Path(extracted_feature_file)
        assert self.path_to_features.exists(), f"Feature file does not exist in {self.path_to_features.absolute().as_posix()}"
        self.features_raw = [data for data in jsonlines.Reader(io.open(self.path_to_features))]
        self.features_flattened = [QueryMatcher.flatten_feature_dict(feature_set) for feature_set in self.features_raw]
        self.features_df = pd.DataFrame(self.features_flattened).set_index('name').drop(columns="timestamp").drop(
            columns="label")
        self.features_column_names = list(self.features_df.columns)
        self.features_list_of_list = [QueryMatcher.prepare_for_matching(feature_set) for feature_set in
                                      self.features_raw]
        self.full_mat = []

    @staticmethod
    def init_from_query_mesh_features(feature_dict):
        features_flattened = [QueryMatcher.flatten_feature_dict(feature_set) for feature_set in feature_dict]
        init_features_df = pd.DataFrame(features_flattened)
        init_features_df = init_features_df.set_index('name').drop(columns="timestamp")
        return init_features_df

    @staticmethod
    def perform_knn(dataset, query, k):
        neighbors = NearestNeighbors(n_neighbors=k).fit(query)
        return neighbors.kneighbors(dataset)

    def compare_features_with_database(self, feature_set,
                                       weights, k=5,
                                       hist_dist_func=None,
                                       scalar_dist_func=None,
                                       n_scalar_features=6):

        scalar_dist_func = QueryMatcher.cosine_distance if not scalar_dist_func else scalar_dist_func
        hist_dist_func = QueryMatcher.cosine_distance if not hist_dist_func else hist_dist_func

        # Make order consistent with matching features db and flatten its distributional values
        feature_dict_in_correct_order = self.prepare_single_feature_for_comparison(feature_set,
                                                                                   list(feature_set.columns))
        # Make an array of the flattened list and reshape so to be [1,]
        feature_instance_vector = np.array(list(feature_dict_in_correct_order.values())).reshape(1, -1)
        # Get processed features from json file
        feature_database_matrix = self.features_df.values
        # Create a matrix of query feature plus all other in db
        self.full_mat = np.vstack((feature_instance_vector, feature_database_matrix))

        # Standardise (zscore)
        scalar_values = self.full_mat[:, :n_scalar_features]
        scalars_mean = np.mean(self.full_mat[:, :n_scalar_features], axis=0)
        scalars_std = np.std(self.full_mat[:, :n_scalar_features], axis=0)
        scalar_values = (scalar_values - scalars_mean) / scalars_std

        # Extract hist values
        hist_values = self.full_mat[:, n_scalar_features:]

        # ttsne = TsneVisualiser()
        # ttsne.raw_data = self.features_raw
        # ttsne.distances = full_mat
        # ttsne.plot()

        if scalar_dist_func == QueryMatcher.perform_knn:
            # Perform knn and store results
            distance_values, indices = self.perform_knn(self.full_mat[0, :].reshape(1, -1), self.full_mat[1:, :], k)
        else:
            # Get results from distance function providing first row of matrix (query) and all others to match it with
            scalar_result = scalar_dist_func(scalar_values[0, :].reshape(1, -1), scalar_values[1:, :]) * weights[0]

            hist_result = hist_dist_func(hist_values[0, :].reshape(1, -1), hist_values[1:, :]) * weights[1]

            result = np.sum(np.vstack((scalar_result, hist_result)), axis=0).reshape(-1, 1)

            # Get indices and values, but indices are not of the filename in db (i.e. 'm + index' won't work)
            indices, distance_values = QueryMatcher.get_top_k(result, k)

        # Retrieve the actual name from indexing the raw dict of shapes
        selected_shapes = np.array(self.features_raw)[indices]
        # For each shape selected, append name and return it
        names = [s["name"] for s in selected_shapes.reshape(-1, )]
        return names, distance_values

    def match_with_db(self, feature_set, k=5, distance_functions=None, weights=None):
        if distance_functions is None:
            distance_functions = []
        feature_set_transformed = QueryMatcher.prepare_for_matching(feature_set=feature_set)
        assert len(feature_set_transformed) == len(
            distance_functions), f"Not enough OR too many distance functions supplied!"

        standardised_features_list_of_list, feature_set_transformed[0] = self.standardize(self.features_list_of_list,
                                                                                          feature_set_transformed,
                                                                                          self.scaler)
        all_distances = np.array(
            [QueryMatcher.mono_run_functions_pipeline(feature_set_transformed, mesh_in_db, distance_functions, weights)
             for mesh_in_db in standardised_features_list_of_list])
        position_in_rank = np.argsort(all_distances)[:k]
        names = [mesh_in_db["name"] for mesh_in_db in np.array(self.features_raw)[position_in_rank]]
        values = all_distances[position_in_rank]
        labels = [mesh_in_db["label"] for mesh_in_db in np.array(self.features_raw)[position_in_rank]]
        print(tuple(zip(names, labels, values)))
        return names, values

    @staticmethod
    def standardize(features_list_of_list, feature_set, scaler):
        """
        Standardisation applied over list of lists as well as query.
        :param features_list_of_list: features_list_of_list: list of lists of array of all normalised features
        :param feature_set: feature_set: query feature
        :param scaler: any scaler from sklearn.preprocessing
        :return: standardised query and list of lists
        """
        features_arr_of_arr = np.array(features_list_of_list)
        scalars = features_arr_of_arr[:, 0]
        reshaped_scalars = np.array([val for arr in scalars for val in arr]).reshape(-1, 8)
        list_standardized_scalars = [x for x in scaler.fit_transform(reshaped_scalars)]
        end_df = pd.DataFrame(features_list_of_list)
        end_df[0] = pd.Series(list_standardized_scalars)
        standardised_features_list_of_list = list(end_df.to_numpy())
        standardised_feature_set = scaler.transform(feature_set[0].reshape(1, -1))
        return standardised_features_list_of_list, standardised_feature_set

    @staticmethod
    def mono_run_functions_pipeline(a_features, b_features, dist_funcs, weights=None):
        """
        Runs the pipeline of functions and computes a comined value
        :param a_features: First mesh feature set
        :param b_features: Second mesh feature set
        :param dist_funcs: List of distance functions
        :param weights: weights for which each distance functions takes part
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
        scalar_features = [np.array([v for k, v in feature_set.items() if
                                     type(v) not in [np.ndarray, list] and k not in QueryMatcher.IGNORE_COLUMNS])]
        distributional_features = [np.array(v) for v in feature_set.values() if type(v) in [np.ndarray, list]]
        return scalar_features + distributional_features

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
        distributional = [{f"{key}_{idx}": val for idx, val in enumerate(dist)} for key, dist in feature_set.items() if
                          type(dist) in [list, np.ndarray]]
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


class TsneVisualiser:
    def __init__(self, raw_data, full_mat, filename):
        self.raw_data = raw_data
        self.full_mat = full_mat
        self.filename = filename

    def plot(self):
        labelled_mat = np.hstack(
            (np.array([dic["label"] for dic in self.raw_data]).reshape(-1, 1), self.full_mat[1:, :]))
        df = pd.DataFrame(data=labelled_mat[:, 1:],
                          index=labelled_mat[:, 0])

        lbl_list = list(df.index)
        color_map = rand_cmap(len(lbl_list), first_color_black=False, last_color_black=True)
        lbl_to_idx_map = dict(zip(lbl_list, range(len(lbl_list))))
        labels = [lbl_to_idx_map[i] for i in lbl_list]

        # Playing around with parameters, this seems like a good fit
        tsne_results = TSNE(perplexity=50, n_iter=10000, learning_rate=500).fit_transform(df.values)
        t_x, t_y = tsne_results[:, 0], tsne_results[:, 1]
        plt.scatter(t_x, t_y, c=labels, cmap=color_map, vmin=0, vmax=len(lbl_list), label=lbl_list, s=10)
        plt.savefig(self.filename, bbox_inches='tight', dpi=200)

    def file_exist(self):
        if os.path.isfile(self.filename):
            return True
        return False

if __name__ == "__main__":
    qm = QueryMatcher(FEATURE_DATA_FILE)
    sampled_mesh = qm.features_flattened[0]
    close_meshes, computed_values = qm.compare_features_with_database(pd.DataFrame(sampled_mesh, index=[0]), 5,
                                                                      QueryMatcher.cosine_distance)
    assert sampled_mesh["name"] in close_meshes
    function_pipeline = [cosine] + ([wasserstein_distance] * (len(qm.features_list_of_list[0]) - 1))
    print(QueryMatcher.mono_run_functions_pipeline(qm.features_list_of_list[0], qm.features_list_of_list[1],
                                                   function_pipeline))
    print(qm.match_with_db(qm.features_raw[0], 5, function_pipeline))
    print("Everything worked!")

    data = DataSet.mono_run_pipeline(DataSet._extract_descr('./processed_data_bkp/bicycle/m1475.ply'))
    normed_data = Normalizer.mono_run_pipeline(data)
    normed_mesh = pv.PolyData(normed_data["history"][-1]["data"]["vertices"],
                              normed_data["history"][-1]["data"]["faces"])
    normed_data['poly_data'] = normed_mesh
    features_dict = FeatureExtractor.mono_run_pipeline(normed_data)
    feature_formatted_keys = [form_key.replace("_", " ").title() for form_key in features_dict.keys()]
    features_df = pd.DataFrame({'key': list(feature_formatted_keys), 'value': list(
        [list(f) if isinstance(f, np.ndarray) else f for f in features_dict.values()])})
    print(qm.match_with_db(features_dict, 5, function_pipeline))
