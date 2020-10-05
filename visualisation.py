# %%
from feature_extractor import FeatureExtractor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.tri as mtri
from reader import DataSet

def plot_mesh(mesh, ax):
    points = mesh.points
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    faces = DataSet._get_cells(mesh)
    return ax.plot_trisurf(X, Y, Z=Z, triangles=faces)


def visualize_histogram(extractor, function_name, item_ids=[0, 1], names=None):
    feature_extraction_function = getattr(extractor, function_name)
    names = names if names else [data["meta_data"]["label"] for data in np.array(extractor.full_data)[item_ids]]
    result_sets = [(data, list(feature_extraction_function(data).values())[0]) for data in np.array(extractor.full_data)[item_ids]]
    num_items = len(result_sets)
    num_bins = extractor.number_bins
    fig = plt.figure(figsize=(5 * num_items, 8))
    axes = [(fig.add_subplot(2, num_items, idx + 1), fig.add_subplot(2, num_items, num_items + idx + 1, projection='3d')) for idx in range(num_items)]
    for (hist_ax, mesh_ax), (data, results), name in zip(axes, result_sets, names):
        hist_ax.set_title(name)
        hist_ax.bar(np.linspace(0, 1, num_bins), results, 1 / num_bins, align='center')
        plot_mesh(data["poly_data"], mesh_ax)
    # fig.suptitle(result_sets[0][0], fontsize=16)
    fig.tight_layout()
    return fig.show()
# #%%
# FE = FeatureExtractor()
# plot_names = "Ant Human Guitar1 Guitar2".split()
# %%
visualize_histogram(FE, "cube_root_volume_four_rand_verts", list(range(4)), plot_names)
visualize_histogram(FE, "angle_three_rand_verts", list(range(4)), plot_names)

# %%
