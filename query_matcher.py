import io
import os
from collections import ChainMap
from collections import OrderedDict
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from sklearn.preprocessing import StandardScaler

from feature_extractor import FeatureExtractor
from helper.config import FEATURE_DATA_FILE


# TODO: EMD does not work for scalars


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

    @staticmethod
    def init_from_query_mesh_features(feature_dict):
        features_flattened = [QueryMatcher.flatten_feature_dict(feature_set) for feature_set in feature_dict]
        init_features_df = pd.DataFrame(features_flattened)
        init_features_df = init_features_df.set_index('name').drop(columns="timestamp")
        return init_features_df

    @staticmethod
    def perform_knn(data_matrix, query, k):
        t = AnnoyIndex(len(query), 'angular')
        if not os.path.isfile('shapes.ann'):
            [t.add_item(idx, data_matrix[idx, :]) for idx in range(data_matrix.shape[0])]
            t.build(10)
            t.save('shapes.ann')
        else:
            t.load('shapes.ann')
        indices, values = t.get_nns_by_vector(query, k, include_distances=True)
        return values, indices

    def match_with_db(self, feature_set, k=5, distance_functions=[], weights=None):
        feature_set_transformed = QueryMatcher.prepare_for_matching(feature_set=feature_set)
        assert len(feature_set_transformed) == len(
            distance_functions), f"Not enough OR too many distance functions supplied!"

        standardised_features_list_of_list, feature_set_transformed, full_standardised_mat, flat_standardised_query = self.standardize(
            self.features_list_of_list,
            feature_set_transformed,
            self.scaler)

        if QueryMatcher.perform_knn in distance_functions:
            values, position_in_rank = self.perform_knn(full_standardised_mat, flat_standardised_query, k)
        else:
            all_distances = np.array(
                [QueryMatcher.mono_run_functions_pipeline(feature_set_transformed, mesh_in_db,
                                                          distance_functions, weights)
                 for mesh_in_db in standardised_features_list_of_list])
            position_in_rank = np.argsort(all_distances)[:k]
            values = all_distances[position_in_rank]

        names = [mesh_in_db["name"] for mesh_in_db in np.array(self.features_raw)[position_in_rank]]
        labels = [mesh_in_db["label"] for mesh_in_db in np.array(self.features_raw)[position_in_rank]]
        print(tuple(zip(names, labels, values)))
        return names, values

    @staticmethod
    def standardize(features_list_of_list, feature_set, scaler):
        """
        Standardisation applied over list of lists as well as query.
        :param full_normed_mat: if true will return the full normalised matrix as lat element
        :param features_list_of_list: features_list_of_list: list of lists of array of all normalised features
        :param feature_set: feature_set: query feature
        :param scaler: any scaler from sklearn.preprocessing
        :return: standardised query and list of lists
        """
        features_arr_of_arr = np.array(features_list_of_list)
        flat_query = [val for sublist in feature_set for val in sublist]
        full_mat = np.array([val for sublist in features_arr_of_arr.flatten() for val in sublist]).reshape(-1,
                                                                                                           len(
                                                                                                               flat_query))
        scalars = full_mat[:, :len(FeatureExtractor.get_pipeline_functions()[0])]
        list_standardized_scalars = [x for x in scaler.fit_transform(scalars)]
        end_df = pd.DataFrame(features_list_of_list)
        end_df[0] = pd.Series(list_standardized_scalars)
        standardised_features_list_of_list = list(end_df.to_numpy())
        standardised_feature_set_scalars = scaler.transform(feature_set[0].reshape(1, -1))
        feature_set[0] = standardised_feature_set_scalars

        # flat_standardised_feature_set_scalars = [val for sublist in standardised_feature_set for val in sublist]
        del flat_query[:len(FeatureExtractor.get_pipeline_functions()[0])]
        flat_standard_query = list(standardised_feature_set_scalars.flatten())
        flat_standard_query.extend(flat_query)

        standardised_features_arr_of_arr = np.array(standardised_features_list_of_list)
        full_mat = np.array(
            [val for sublist in standardised_features_arr_of_arr.flatten() for val in sublist]).reshape(-1,
                                                                                                        len(
                                                                                                            flat_standard_query))
        return standardised_features_list_of_list, feature_set, full_mat, flat_standard_query

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
