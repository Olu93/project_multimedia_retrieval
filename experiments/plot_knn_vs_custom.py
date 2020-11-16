# %%
import os
from collections import Counter
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyvista as pv
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance

from feature_extractor import FeatureExtractor
from helper.config import DATA_PATH_PSB
from helper.config import FEATURE_DATA_FILE
from helper.misc import get_sizes_features
from normalizer import Normalizer
from query_matcher import QueryMatcher
from reader import DataSet


def plot_comparison(sample_labels, distance):
    qm = QueryMatcher(FEATURE_DATA_FILE)
    labelled_occurences = tuple(zip(sample_labels,
                                    [Counter(pd.DataFrame(qm.features_flattened)["label"]).get(lbl) for lbl in
                                     sample_labels]))
    names = [[f for f in qm.features_raw if f["label"] == lbl][0]["name"] for lbl in sample_labels]
    sampled_labelled = dict(zip(labelled_occurences, names))
    paths = []

    for path, subdirs, files in os.walk(DATA_PATH_PSB):
        for name in files:
            if ("off" or "ply") in name:
                paths.append(os.path.join(path, name))

    n_singletons, n_distributionals, mapping_of_labels = get_sizes_features(with_labels=True)

    n_hist = len([key for key, val in mapping_of_labels.items() if "hist_" in key])
    n_skeleton = len([key for key, val in mapping_of_labels.items() if "skeleton_" in key])

    if distance != "knn":
        # Custom
        weights = ([3]) + \
                  ([100] * n_hist) + \
                  ([1] * n_skeleton)

        function_pipeline = [cosine] + \
                            ([wasserstein_distance] * n_hist) + \
                            ([wasserstein_distance] * n_skeleton)
    else:
        # KNN
        weights = ([1]) + \
                  ([1] * n_hist) + \
                  ([1] * n_skeleton)

        function_pipeline = [QueryMatcher.perform_knn] + (
                [QueryMatcher.perform_knn] * n_distributionals)

    normalizer = Normalizer()
    out_dict = defaultdict(list)
    for info_tuple, mesh_idx in sampled_labelled.items():
        full_path = [p for p in paths if mesh_idx in p][0]
        print(f"Processing: {full_path}")
        mesh = DataSet._read(Path(full_path))
        normed_data = normalizer.mono_run_pipeline(mesh)
        normed_mesh = pv.PolyData(normed_data["history"][-1]["data"]["vertices"],
                                  normed_data["history"][-1]["data"]["faces"])
        normed_data['poly_data'] = normed_mesh

        features_dict = FeatureExtractor.mono_run_pipeline_old(normed_data)

        indices, distance_values, _ = qm.match_with_db(features_dict,
                                                       k=10,
                                                       distance_functions=function_pipeline,
                                                       weights=weights)
        if mesh_idx in indices:
            idx_of_idx = indices.index(mesh_idx)
            indices.remove(mesh_idx)
            del distance_values[idx_of_idx]
            distance_values.insert(0, 0)

        indices = indices[4:]
        indices.insert(0, mesh_idx)
        distance_values = distance_values[5:]
        out_dict[info_tuple].append({mesh_idx: (indices, distance_values)})
        print(out_dict)

    class_idx = 0
    plt = pv.Plotter(off_screen=True, shape=(6, 5))
    for key, val in out_dict.items():
        print(class_idx)
        for v in val:
            el_idx = 0
            distances = list(list(v.values())[0][1])
            for name, dist in zip(list(v.values())[0][0], distances):
                print(el_idx)
                plt.subplot(class_idx, el_idx)

                full_path = [p for p in paths if name in p][0]

                mesh = DataSet._read(Path(full_path))
                curr_mesh = pv.PolyData(mesh["data"]["vertices"], mesh["data"]["faces"])
                plt.add_mesh(curr_mesh, color='r')
                plt.reset_camera()
                plt.view_isometric()
                if el_idx != 0:
                    plt.add_text(f"{el_idx} - Dist: {round(dist,4)}", font_size=20)
                elif el_idx == 0 and class_idx == 0:
                    plt.add_text(
                        f"             Query\nClass: {key[0].replace('_', ' ').title()}" + f"\nInstances: {key[1]}",
                        font_size=20)
                else:
                    plt.add_text(f"Class: {key[0].replace('_', ' ').title()}" + f"\nInstances: {key[1]}", font_size=20)

                el_idx += 1
        class_idx += 1

    if distance == "knn":
        plt.screenshot(f"fig\\comparison_knn.jpg", window_size=(1920, 2160))
    else:
        plt.screenshot(f"figs\\comparison_custom_distance.jpg", window_size=(1920, 2160))


if __name__ == '__main__':
    selected_labels = ["fighter_jet", "human", "potted_plant", "helicopter", "ant", "desk_lamp"]
    plot_comparison(selected_labels, "knn")
