import io
import json

import pandas as pd
from query_matcher import QueryMatcher
from normalizer import Normalizer
import pathlib
from helper.misc import get_feature_type_positions, jsonify, load_feature_data, screenshot_mesh
from matplotlib import pyplot as plt
import numpy as np
from reader import DataSet, PSBDataset
from feature_extractor import FeatureExtractor
from helper.config import CLASS_FILE, DATA_PATH_NORMED_SUBSET, DATA_PATH_PSB, FEATURE_DATA_FILE
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d


def plot_mesh(mesh, ax):
    points = mesh.points
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    faces = DataSet._get_cells(mesh)
    return ax.plot_trisurf(X, Y, Z=Z, triangles=faces)


def compute_stats(features_df, feature_names, lbl_name="label") -> pd.DataFrame:
    agg_df_mean = features_df.groupby(lbl_name).mean()
    agg_df_std = features_df.groupby(lbl_name).std()
    output_dict = {}
    for feature in feature_names:
        subset_mean = agg_df_mean.filter(regex=f"^{feature}", axis=1)
        subset_std = agg_df_std.filter(regex=f"^{feature}", axis=1)
        output_dict[feature] = {"mean": subset_mean, "std": subset_std}

    return output_dict


def visualize_histograms(mesh_features, data=None, names=None, plot_titles=None, is_coarse=False, example_title="Class Example"):
    cols = get_feature_type_positions(list(mesh_features[0].keys()))
    feature_names = list(cols["hist"].keys())[:-2]
    # class_memberships = PSBDataset.load_classes(CLASS_FILE if )
    meshes = [DataSet._read(mesh["file_path"]) for mesh in mesh_features]

    lbl_name = "label" if not is_coarse else "label_coarse"
    cols_2_keep = [lbl_name, "name"] + feature_names
    ignore_cols = set(QueryMatcher.IGNORE_COLUMNS) - set([lbl_name])
    features_flattened = [QueryMatcher.flatten_feature_dict(feature_set, cols_2_keep) for feature_set in data]
    features_df = pd.DataFrame(features_flattened).set_index("name").drop(columns=ignore_cols, errors='ignore')

    labels = [features_df.loc[mesh["meta_data"]["name"]]["label"] for mesh in meshes]
    names = names if names else labels

    new_result_sets = [{fn: item.get(fn) for fn in feature_names} for item in mesh_features]
    plot_titles = plot_titles if plot_titles else feature_names
    result_sets = mesh_features
    num_items = len(feature_names)
    num_rows = len(result_sets)
    fontsize = 15
    # Compute data means
    data_stats = compute_stats(features_df, feature_names)

    # Start plotting
    fig = plt.figure(figsize=(4 * num_items, 3 * num_rows))
    fig.patch.set_visible(False)
    num_cols = num_items + 1
    hist_axes = fig.subplots(num_rows, num_cols)
    for idx, (hist_ax, result_set) in enumerate(zip(hist_axes, new_result_sets)):
        for ax, (title, results) in zip(hist_ax[:len(feature_names)], result_set.items()):

            x = np.linspace(0, 1, len(results))
            y = np.array(results)
            xnew = np.linspace(x.min(), x.max(), 300)
            spl = interp1d(x, y, kind='cubic')
            ynew = spl(xnew)
            ax.plot(xnew, ynew, c="red", label="Current Example")

            class_mean = data_stats[title]["mean"].loc[labels[idx]] if title in data_stats else []
            class_std = data_stats[title]["std"].loc[labels[idx]] if title in data_stats else []
            ax.plot(x, class_mean.values, c="b", label=f"Class $\\mu$ \\& $\\pm2\\sigma$ confidence interval")
            ax.fill_between(x, class_mean - 2 * class_std, class_mean + 2 * class_std, color='b', alpha=.1)

        ax = fig.add_subplot(num_rows, num_cols, num_cols + (idx * num_cols))
        img = screenshot_mesh(meshes[idx]["poly_data"])
        ax.imshow(img, aspect='auto')
        ax.set_title(names[idx].replace("_", " ").title(), loc="center", color="white", y=.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.autoscale_view('tight')
        ax.tick_params(axis='both', which='both', length=0)

    for ax in hist_axes[:, -1].flatten():
        ax.set_yticks([])
        ax.set_xticks([])
    hist_axes[0, 0].get_shared_y_axes().join(*hist_axes[:, :-1].flatten().tolist())

    for ax in hist_axes[:, 1:-1].flatten():
        ax.set_yticks([])

    for ax_col, x_title in zip(hist_axes[0, :], plot_titles):
        ax_col.set_title(x_title, fontsize=fontsize)
    hist_axes[0, -1].set_title(example_title, fontsize=fontsize)

    handles, labels = hist_axes[-1, -2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=len(labels),
        bbox_to_anchor=(.5, 0.01),
        fontsize=fontsize,
    )
    fig.suptitle("Distributional features", fontsize=fontsize)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig


if __name__ == "__main__":

    fname = "trash/mesh_features_for_fig_variant1.json"
    if pathlib.Path(fname).exists():
        mesh_features = json.load(io.open(fname))
    else:
        path_to_data = pathlib.Path(DATA_PATH_PSB)
        path_to_data = pathlib.Path("trash")
        mesh_files = [
            path_to_data / "m18.ply",  #"0/m18/m18.off",
            path_to_data / "m118.ply",  #"1/m118/m118.off",
            path_to_data / "m1.ply",  #"0/m1/m1.off",
        ]
        meshes = [DataSet._read(fp) for fp in mesh_files]
        # meshes = [Normalizer.mono_run_pipeline(mesh) for mesh in meshes]
        mesh_features = [FeatureExtractor.mono_run_pipeline(mesh) for mesh in meshes]
        mesh_features = [dict(file_path=(path_to_data / (mesh["name"] + ".ply")).absolute().as_posix(), **mesh) for mesh in mesh_features]
        json.dump(list(map(jsonify, mesh_features)), io.open(fname, "w"))

    mesh_names = "Spider Human Ant Guitar2".split()
    data = load_feature_data(FEATURE_DATA_FILE)
    ptitles = "D4 A3 D2 D1 D3".split()
    fig = visualize_histograms(mesh_features, data=data, plot_titles=ptitles)
    fig.savefig('figs/fig_feature_distribution_variant_1.png')
    plt.show()