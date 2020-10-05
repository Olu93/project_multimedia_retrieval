# %%
from feature_extractor import FeatureExtractor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.tri as mtri

#%%
FE = FeatureExtractor()

# %%
example1 = FE.full_data[0]
example2 = FE.full_data[3]
result1 = list(FE.cube_root_volume_four_rand_verts(example1).values())[0]
result2 = list(FE.cube_root_volume_four_rand_verts(example2).values())[0]
ex_points1 = example1["poly_data"].points
ex_points2 = example2["poly_data"].points
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
from reader import DataSet

subsample_factor = 1
X, Y, Z = ex_points1[::subsample_factor, 0], ex_points1[::subsample_factor, 1], ex_points1[::subsample_factor, 2]
faces = DataSet._get_cells(example1["poly_data"])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(X, Y, Z=Z, triangles=faces)

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

assert np.all(x[:,0,0] == x_)
assert np.all(y[0,:,0] == y_)
assert np.all(z[0,0,:] == z_)