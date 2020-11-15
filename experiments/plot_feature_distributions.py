import io
import json
from normalizer import Normalizer
import pathlib
from helper.misc import get_feature_type_positions, jsonify, screenshot_mesh
from matplotlib import pyplot as plt
import numpy as np
from reader import DataSet, PSBDataset
from feature_extractor import FeatureExtractor
from helper.config import DATA_PATH_NORMED_SUBSET, DATA_PATH_PSB


def plot_mesh(mesh, ax):
    points = mesh.points
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    faces = DataSet._get_cells(mesh)
    return ax.plot_trisurf(X, Y, Z=Z, triangles=faces)


def visualize_histograms(mesh_features, selection, item_ids=[0, 1], names=None, plot_titles=None):
    cols = get_feature_type_positions(list(mesh_features[0].keys()))
    feature_names = list(cols["hist"].keys())
    meshes = [DataSet._read(mesh["file_path"]) for mesh in mesh_features]
    names = names if names else [data["meta_data"]["label"] for data in meshes]
    new_result_sets = [{feature_names[idx]: item.get(feature_names[idx]) for idx in selection} for item in mesh_features]
    plot_titles = plot_titles if plot_titles else list(np.array(feature_names)[selection])
    result_sets = mesh_features
    num_items = len(plot_titles)
    num_rows = len(result_sets)
    num_bins = len(mesh_features[0].get(feature_names[0]))
    fig = plt.figure(figsize=(4 * num_items, 3 * num_rows))
    fig.patch.set_visible(False)
    num_cols = num_items + 1
    # hist_axes = fig.subplots(num_rows, num_cols, sharex=True, sharey=True)
    hist_axes = fig.subplots(num_rows, num_cols)

    for idx, (hist_ax, result_set) in enumerate(zip(hist_axes, new_result_sets)):
        for ax, (title, results) in zip(hist_ax[:len(selection)], result_set.items()):
            ax.bar(np.linspace(0, 1, len(results)), results, align='center', width=.1, edgecolor=None)
            # ax.set_xticks([])
            # ax.set_yticks([])
        ax = fig.add_subplot(num_rows, num_cols, num_cols + (idx * num_cols))
        # ax.remove()
        img = screenshot_mesh(meshes[idx]["poly_data"])
        ax.imshow(img, aspect='auto')
        # ax.axis("off")
        ax.set_title(names[idx], loc="center", color="white", y=.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.autoscale_view('tight')
        ax.tick_params(axis='both', which='both', length=0)

    # for idx, (name, ax, mesh) in enumerate(zip(names, hist_axes[-1, :], meshes)):
    #     last_index = (num_rows * num_items) + idx + 1
    #     ax = fig.add_subplot(num_rows + 1, num_items, last_index)
    #     img = screenshot_mesh(mesh["poly_data"])
    #     ax.imshow(img)

    for ax in hist_axes[:, -1].flatten():
        # ax_row.set_ylabel(str(y_title), rotation=90)
        ax.set_yticks([])
        ax.set_xticks([])
    hist_axes[0, 0].get_shared_y_axes().join(*hist_axes[:, :-1].flatten().tolist())

    for ax in hist_axes[:, 1:-1].flatten():
        ax.set_yticks([])

    for ax in hist_axes[:-1, :-1].flatten():
        ax.set_xticks([])
        # ax.set_xticks([])

    for ax_col, x_title in zip(hist_axes[0, :], plot_titles):
        ax_col.set_title(x_title)
    hist_axes[0, -1].set_title("Class")

    fig.tight_layout()
    # plt.show()
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
    fig = visualize_histograms(mesh_features, [0, 1, 2, 3], item_ids=list(range(4)), names=mesh_names)
    plt.show()
    # fig.savefig('fig/feature_distribution_variant_1.png')