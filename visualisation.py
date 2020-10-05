# %%
from feature_extractor import FeatureExtractor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.tri as mtri
from reader import DataSet

#%%
FE = FeatureExtractor()
# %%
example1 = FE.full_data[0]
# example2 = FE.full_data[1]
result1 = list(FE.cube_root_volume_four_rand_verts(example1).values())[0]
# plt.hist(result1, bins=np.linspace(0, 1, 10))
plt.bar(np.linspace(0, 1, 10), result1, .1, align='center')
result1


# %%
def plot_mesh(mesh, ax):
    points = mesh.points
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    faces = DataSet._get_cells(mesh)
    return ax.plot_trisurf(X, Y, Z=Z, triangles=faces)


def visualize_histogram(extractor, feature_extraction_function, item_ids=[0, 1]):
    result_sets = [(data, list(feature_extraction_function(data).values())[0]) for data in np.array(extractor.full_data)[item_ids]]
    num_items = len(result_sets)
    num_bins = FE.number_bins
    fig = plt.figure(figsize=(5 * num_items, 8))
    axes = [(fig.add_subplot(2, num_items, idx + 1), fig.add_subplot(2, num_items, num_items + idx + 1, projection='3d')) for idx in range(num_items)]
    for (hist_ax, mesh_ax), (data, results) in zip(axes, result_sets):
        hist_ax.bar(np.linspace(0, 1, num_bins), results, 1 / num_bins, align='center')
        plot_mesh(data["poly_data"], mesh_ax)
    fig.tight_layout()
    return fig.show()


visualize_histogram(FE, FE.cube_root_volume_four_rand_verts, list(range(4)))
# %%

# result2 = list(FE.cube_root_volume_four_rand_verts(example2).values())[0]
# axes = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223, projection='3d'), fig.add_subplot(224, projection='3d')
# fig, axes = plt.subplots(2, 2, subplot_kw=dict(projection='3d'))
# axes[0].plot(result1)
# axes[1].plot(result2)
# plot_mesh(example1["poly_data"], axes[2])
# plot_mesh(example2["poly_data"], axes[3])
# %%
# fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection='3d'))
fig = plt.figure()
fig.add_subplot(2, 2, 1)
fig.add_subplot(2, 2, 3)
plt.tight_layout()

# %%

subsample_factor = 10
X, Y, Z = np.meshgrid(ex_points1[::subsample_factor, 0], ex_points1[::subsample_factor, 1], ex_points1[::subsample_factor, 2])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z)

# %%
X, Y, Z = np.meshgrid(ex_points1[::subsample_factor, 0], ex_points1[::subsample_factor, 1], ex_points1[::subsample_factor, 2])
points = np.c_[X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]
foo = pv.PolyData(points)

# %%

# %%

# %%
import pyvista as pv
X, Y, Z = ex_points1[::subsample_factor, 0], ex_points1[::subsample_factor, 1], ex_points1[::subsample_factor, 2]
grid = pv.StructuredGrid(X, Y, Z).me
grid.plot()
# %%
X, Y = np.meshgrid(ex_points1[::subsample_factor, 0], ex_points1[::subsample_factor, 1])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, ex_points1[::subsample_factor, 2])
ax = fig.add_subplot(111, projection='3d')

# %%
subsample_factor = 10
X, Y, Z = ex_points1[::subsample_factor, 0], ex_points1[::subsample_factor, 1], ex_points1[::subsample_factor, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z)
# %%
import numpy as np

x_ = np.linspace(0., 1., 10000)[::100]
y_ = np.linspace(1., 2., 10000)[::100]
z_ = np.linspace(3., 4., 10000)[::100]

x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

assert np.all(x[:, 0, 0] == x_)
assert np.all(y[0, :, 0] == y_)
assert np.all(z[0, 0, :] == z_)
