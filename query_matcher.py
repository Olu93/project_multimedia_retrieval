import io
import os
from collections import ChainMap
from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import jsonlines
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from sklearn.preprocessing import StandardScaler
import itertools
from feature_extractor import FeatureExtractor
from helper.config import DEBUG, FEATURE_DATA_FILE
from helper.misc import get_feature_type_positions, get_sizes_features

# TODO: EMD does not work for scalars
# if DEBUG:
#     debug_file = io.open("analysis.txt", mode="w")


class QueryMatcher(object):
    IGNORE_COLUMNS = ["timestamp", "label", "label_coarse"]
    CONST_SCALAR_ALL_COL = "scalar_all"

    def __init__(self, extracted_feature_file, label_coarse=False):
        self.scaler = StandardScaler()
        self.path_to_features = Path(extracted_feature_file)
        assert self.path_to_features.exists(), f"Feature file does not exist in {self.path_to_features.absolute().as_posix()}"
        self.features_raw_init = [data for data in jsonlines.Reader(io.open(self.path_to_features))]
        self.features_raw = [dict(data) for data in self.features_raw_init]
        self.class_mapping = {item["name"]: {"label": item["label"], "label_coarse": item["label_coarse"]} for item in self.features_raw}
        if label_coarse:
            for data in self.features_raw:
                data.update(label=data["label_coarse"])
        self.list_of_list_df = pd.DataFrame(self.features_raw)
        self.list_of_list_cols = np.array(self.list_of_list_df.columns)
        self.col_mapping = get_feature_type_positions(list(self.list_of_list_df.columns))
        self.scalar_cols = self.list_of_list_cols[list(self.col_mapping["scalar"].values())]
        scalers, features_list_of_list_df = QueryMatcher.scale_data(self.list_of_list_df, self.col_mapping)
        feature_list_names, features_list_of_list = list(features_list_of_list_df.columns), features_list_of_list_df.values
        self.scalers = scalers
        self.features_list_names = feature_list_names
        self.features_list_of_list = features_list_of_list
        self.features_flattened = [QueryMatcher.flatten_feature_dict(feature_set, self.list_of_list_cols) for feature_set in self.features_raw]
        self.features_df = pd.DataFrame(self.features_flattened).set_index('name').drop(columns=QueryMatcher.IGNORE_COLUMNS, errors='ignore')
        self.features_df_all_scaled = pd.DataFrame(StandardScaler().fit_transform(self.features_df), columns=self.features_df.columns, index=self.features_df.index)

        self.features_df_subset_scalar = self.features_df.filter(regex="^scalar_", axis=1)
        self.features_df_subset_hist = self.features_df.filter(regex="^hist_", axis=1)
        self.features_df_subset_skeleton = self.features_df.filter(regex="^skeleton_", axis=1)
        self.features_df_properly_scaled = pd.DataFrame(
            np.hstack([
                StandardScaler().fit_transform(self.features_df_subset_scalar),
                self.features_df_subset_hist.values,
                StandardScaler().fit_transform(self.features_df_subset_skeleton),
            ]),
            columns=self.features_df.columns,
            index=self.features_df.index,
        )

        self.features_column_names = list(self.features_df.columns)

    @staticmethod
    def scale_data(data, mapping_cols):
        # skeleton scaling
        skeleton_features = [(col, np.array([row for row in data[col]])) for col in mapping_cols["skeleton"].keys()]
        skeleton_flatten_features = [(col, feature_data.reshape(-1, 1), feature_data.shape) for col, feature_data in skeleton_features]
        skeleton_fit_scalers = [(col, StandardScaler().fit(feature_data), feature_data, former_shape) for col, feature_data, former_shape in skeleton_flatten_features]
        skeleton_scale_values = [(col, scaler, scaler.transform(feature_data).reshape(former_shape)) for col, scaler, feature_data, former_shape in skeleton_fit_scalers]
        skeleton_data = [(col, skeleton_data) for col, _, skeleton_data in skeleton_scale_values]

        # scalar scaling
        scalar_features = [(QueryMatcher.CONST_SCALAR_ALL_COL, data[list(mapping_cols["scalar"].keys())])]
        scalar_fit_scalers = [(col, StandardScaler().fit(feature_data), feature_data) for col, feature_data in scalar_features]
        scalar_scale_values = [(col, scaler, scaler.transform(feature_data)) for col, scaler, feature_data in scalar_fit_scalers]
        scalar_data = [(col, singular_data) for col, _, singular_data in scalar_scale_values]

        # hist non-scaling
        hist_data = [(col, np.array([row for row in data[col]])) for col in mapping_cols["hist"].keys()]

        scaler_dictionary = dict([(col, scaler) for col, scaler, _ in scalar_scale_values] + [(col, scaler) for col, scaler, _ in skeleton_scale_values])
        features_list_of_list = scalar_data + hist_data + skeleton_data
        header_lol = [val[0] for val in features_list_of_list]
        transposed_lol = list(zip(*([val[1] for val in features_list_of_list])))
        list_of_list_table = pd.DataFrame(transposed_lol, columns=header_lol)
        return scaler_dictionary, list_of_list_table

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

    def map_to_label(self, name, is_coarse):
        return self.class_mapping[name].get("label" if not is_coarse else "label_coarse", None)

    def compare_features_with_database(self, feature_set, weights, k=5, hist_dist_func=None, scalar_dist_func=None, n_scalar_features=6):

        scalar_dist_func = QueryMatcher.cosine_distance if not scalar_dist_func else scalar_dist_func
        hist_dist_func = QueryMatcher.cosine_distance if not hist_dist_func else hist_dist_func

        # Make order consistent with matching features db and flatten its distributional values
        feature_dict_in_correct_order = self.prepare_single_feature_for_comparison(feature_set, list(feature_set.columns))
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
        labels = [mesh_in_db["label"] for mesh_in_db in np.array(self.features_raw)[indices]]
        return names, distance_values, labels

    def match_with_db(self, feature_set, k=5, distance_functions=[], weights=None):
        # feature_set_transformed = QueryMatcher.prepare_for_matching(feature_set=feature_set)
        standardised_item = QueryMatcher.prepare_for_matching(feature_set, self.scalers, self.col_mapping, self.features_list_names)
        # print(weights)
        # print(standardised_item[-1])
        len_fst = len(standardised_item)
        len_df = len(distance_functions)
        assert len_fst == len_df, f"Not enough OR too many distance functions supplied! - requires {len_fst} functions and not {len_df}"

        if QueryMatcher.perform_knn in distance_functions:
            values, position_in_rank = self.perform_knn(QueryMatcher.flatten_feature_dict(standardised_item), self.features_flattened, k)
        else:
            all_distances = np.array(
                [QueryMatcher.mono_run_functions_pipeline(standardised_item, mesh_in_db, distance_functions, weights) for mesh_in_db in self.features_list_of_list])
            position_in_rank = np.argsort(all_distances)[:k]
            values = all_distances[position_in_rank]

        values = list(values)
        names = [mesh_in_db["name"] for mesh_in_db in np.array(self.features_raw)[position_in_rank]]
        labels = [mesh_in_db["label"] for mesh_in_db in np.array(self.features_raw)[position_in_rank]]
        # print(tuple(zip(names, labels, values)))
        return names, values, labels

    @staticmethod
    def prepare_for_matching(feature_set, scalers, col_mapping, final_col_order_mapping):
        """
        Standardisation applied over list of lists as well as query.
        :param features_list_of_list: features_list_of_list: list of lists of array of all normalised features
        :param feature_set: feature_set: query feature
        :param scaler: any scaler from sklearn.preprocessing
        :return: standardised query and list of lists
        """
        scalar_features = np.array([feature_set[col_name] for col_name in col_mapping["scalar"].keys()]).reshape(1, -1)
        standardized_scalar_features = {QueryMatcher.CONST_SCALAR_ALL_COL: list(scalers[QueryMatcher.CONST_SCALAR_ALL_COL].transform(scalar_features))}
        standardized_hist_features = {col: np.array(feature_set[col]) for col in col_mapping["hist"].keys()}
        standardized_skeleton_features = {col: scalers[col].transform(np.array(feature_set[col]).reshape(-1, 1)).flatten() for col in col_mapping["skeleton"].keys()}

        # Making sure that the order is correct
        all_combined = OrderedDict(**standardized_scalar_features, **standardized_hist_features, **standardized_skeleton_features)

        pre_output = OrderedDict([(col, None) for col in final_col_order_mapping])
        for col_name, val in all_combined.items():
            pre_output[col_name] = val

        return list(pre_output.values())

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
        results = {f"{fn.__name__}_{idx}": {"distance": w * fn(a, b), "input": (a, b, w)} for idx, (a, b, fn, w) in enumerate(zip(a_features, b_features, dist_funcs, weights))}
        # if DEBUG:
        #     pprint({key: val["distance"] for key, val in results.items()}, stream=debug_file)
        return sum([val["distance"] for _, val in results.items()])

    # @staticmethod
    # def prepare_for_matching(feature_set):
    #     """
    #     Acts as preparation for the matching process, as different features will use different distance functions.

    #     Puts scalar values into a single list.
    #     Every distributional feature will be a single list.
    #     In the all lists are combined into list of lists.
    #     """

    #     f_items = list(feature_set.items())
    #     prepared = {}
    #     prepared["scalar_combined"] = np.array([f_items[position] for position in mapping_of_indices["scalar"]])

    #     distributional_features = {f_items[position][0]: f_items[position][1] for position in mapping_of_indices["hist"]}
    #     skeleton_features = {f_items[position][0]: f_items[position][1] for position in mapping_of_indices["skeleton"]}
    #     prepared.update(distributional_features)
    #     prepared.update(skeleton_features)
    #     return prepared

    @staticmethod
    def flatten_feature_dict(feature_set, l_ordered_feat: list = None):
        if l_ordered_feat is None:
            # for compatibility
            singletons = {key: value for key, value in feature_set.items() if type(value) not in [list, np.ndarray]}
            distributional = [{f"{key}_{idx}": val for idx, val in enumerate(dist)} for key, dist in feature_set.items() if type(dist) in [list, np.ndarray]]
            flattened_feature_set = dict(ChainMap(*distributional, singletons))
            return flattened_feature_set

        flattened_feature_set = {}
        for fname in l_ordered_feat:
            if isinstance(feature_set[fname], (list, np.ndarray)):
                flattened_feature_set.update({f"{fname}_{idx}": val for idx, val in enumerate(feature_set[fname])})
            if not isinstance(feature_set[fname], (list, np.ndarray)):
                flattened_feature_set[fname] = feature_set[fname]
        return flattened_feature_set
        # for feature_name in ordered_list:
        #     flattened_feature_set.extend()

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
    # sampled_mesh = qm.features_flattened[0]
    # close_meshes, computed_values = qm.compare_features_with_database(pd.DataFrame(sampled_mesh, index=[0]), 5, QueryMatcher.cosine_distance)
    # assert sampled_mesh["name"] in close_meshes
    tmp_mappings = get_feature_type_positions(list(qm.list_of_list_cols))
    function_pipeline = [cosine] + ([wasserstein_distance] * (len(tmp_mappings["hist"]))) + ([cityblock] * (len(tmp_mappings["skeleton"])))
    print(QueryMatcher.mono_run_functions_pipeline(qm.features_list_of_list[0], qm.features_list_of_list[1], function_pipeline))
    print(qm.match_with_db(qm.features_raw[0], 5, function_pipeline))
    print("Everything worked!")

    data = DataSet.mono_run_pipeline(DataSet._extract_descr('./processed_data_bkp/bicycle/m1475.ply'))
    normed_data = Normalizer.mono_run_pipeline(data)
    normed_mesh = pv.PolyData(normed_data["history"][-1]["data"]["vertices"], normed_data["history"][-1]["data"]["faces"])
    normed_data['poly_data'] = normed_mesh
    features_dict = FeatureExtractor.mono_run_pipeline_old(normed_data)
    feature_formatted_keys = [form_key.replace("_", " ").title() for form_key in features_dict.keys()]
    features_df = pd.DataFrame({'key': list(feature_formatted_keys), 'value': list([list(f) if isinstance(f, np.ndarray) else f for f in features_dict.values()])})
    print(qm.match_with_db(features_dict, 5, function_pipeline))
