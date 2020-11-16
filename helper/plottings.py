from helper.skeleton import extract_endpoints
import matplotlib.pyplot as plt
import numpy as np
from reader import DataSet


def plot_mesh(mesh, ax):
    points = mesh.points
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    faces = DataSet._get_cells(mesh)
    return ax.plot_trisurf(X, Y, Z=Z, triangles=faces)


def visualize_histograms(extractor, functions, item_ids=[0, 1], names=None, plot_titles=None):
    meshes = np.array(extractor.full_data)[item_ids]
    names = names if names else [data["meta_data"]["label"] for data in meshes]
    plot_titles = plot_titles if plot_titles else list(functions.values())
    result_sets = [[(data, list(func(data).values())[0]) for data in meshes] for func in functions.keys()]
    num_items = len(item_ids)
    num_rows = len(result_sets)
    num_bins = extractor.number_bins
    fig = plt.figure(figsize=(10 * num_items, 9 * num_rows))
    # axes = [
    #     (fig.add_subplot(2, num_items, idx + 1), fig.add_subplot(2, num_items, num_items + idx + 1, projection='3d'))
    #     for idx
    #     in range(num_items)
    #     ]
    hist_axes = fig.subplots(num_rows + 1, num_items, sharex=True)
    #
    for idx, (hist_ax, result_set) in enumerate(zip(hist_axes, result_sets)):

        for ax, (data, results) in zip(hist_ax[:4], result_set):
            ax.bar(np.linspace(0, 1, num_bins), results, 1 / num_bins, align='center')

    for idx, (name, ax, mesh) in enumerate(zip(names, hist_axes[-1, :], meshes)):
        ax.remove()
        last_index = (num_rows * num_items) + idx + 1
        ax = fig.add_subplot(num_rows + 1, num_items, last_index, projection='3d')
        plot_mesh(mesh["poly_data"], ax)

    for ax_row, y_title in zip(hist_axes[:, 0], plot_titles):
        ax_row.set_ylabel(y_title, rotation=90, fontsize=30.0)

    for ax_col, x_title in zip(hist_axes[0, :], names):
        ax_col.set_title(x_title, fontsize=30.0)

    fig.tight_layout()
    # plt.show()
    return fig


def visualize_skeleton_graph_extraction(sillhouette, G, endpoints=True, conjuncts=True, edge_lengths=True):
    def create_base_image(ax, sillhouette, G):
        ax.imshow(sillhouette, cmap='gray')
        for (s, e) in G.edges():
            ps = G[s][e]['pts']
            ax.plot(ps[:, 1], ps[:, 0], 'green')
        return ax

    num = sum([endpoints, conjuncts, edge_lengths])
    fig, axes = plt.subplots(1, num + 1)
    axes_list = list(axes.ravel()[::-1])
    create_base_image(axes_list.pop(), sillhouette, G)
    if endpoints:
        nodes = extract_endpoints(G)
        ax = create_base_image(axes_list.pop(), sillhouette, G)
        ps = np.array([G.nodes()[i]['o'] for i in nodes])
        ax.plot(ps[:, 1], ps[:, 0], 'r.')
    if conjuncts:
        nodes = extract_endpoints(G)
        ax = create_base_image(axes_list.pop(), sillhouette, G)
        ps = np.array([G.nodes()[i]['o'] for i in nodes])
        ax.plot(ps[:, 1], ps[:, 0], 'r.')