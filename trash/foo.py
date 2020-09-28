# %%
import pygmsh
import pyvista as pv
from pyvista import examples
import meshio
from reader import DataSet
import io
import pyacvd
import numpy as np
import pygalmesh
from pyntcloud import PyntCloud
import pandas as pd
import numpy as np
# # %%
# cow = examples.download_face().triangulate()
# cow = cow.subdivide(2)
# cow = cow.decimate_pro(.7)
# polygons = DataSet._get_cells(cow)
# cow.plot(color='w', show_edges=True)

# # %%
# clus = pyacvd.Clustering(cow)
# # mesh is not dense enough for uniform remeshing
# clus.cluster(20000)
# remesh = clus.create_mesh()
# remesh.plot(color='w', show_edges=True)

# # %%
# dec = remesh.decimate_pro(0.7)
# dec.plot(color='w', show_edges=True)
# # %%
# clus = pyacvd.Clustering(dec)
# # mesh is not dense enough for uniform remeshing
# # clus.subdivide(3)
# clus.cluster(20000)
# new_remesh = clus.create_mesh()
# new_remesh.plot(color='w', show_edges=True)

# %%

# %%
# import pygalmesh
# mesh = pygalmesh.remesh_surface(
#     cow,
#     edge_size=0.025,
#     facet_angle=25,
#     facet_size=0.1,
#     facet_distance=0.001,
#     verbose=False,
# )

# %%
# n = 200
# shape = (n, n, n)
# h = [1.0 / s for s in shape]
# vol = np.zeros(shape, dtype=np.uint16)
# i, j, k = np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])
# ii, jj, kk = np.meshgrid(i, j, k)
# vol[ii * ii + jj * jj + kk * kk < n ** 2] = 1
# vol[ii * ii + jj * jj + kk * kk < (0.5 * n) ** 2] = 2

# %%
# points = mesh.points
# faces = mesh.cells[0].data
# num_face_vertices = np.array([len(face) for face in faces]).reshape(-1,1)
# pyvista_faces = np.hstack([num_face_vertices, faces]).flatten()
# poly_data_object = pv.PolyData(points, faces)
# poly_data_object.plot(color='w', show_edges=True)
# %%
# foot = pv.voxelize(pv.read("ant.off"))
foot = examples.download_cow()
voxel_points = pd.DataFrame(foot.points, columns="x y z".split())
voxel_points.head()
# %%
cloud = PyntCloud(voxel_points)
cloud
# %%
n = 64
voxelgrid_id = cloud.add_structure("voxelgrid", n_x=n, n_y=n, n_z=n)
voxelgrid = cloud.structures[voxelgrid_id]
voxelgrid
# %%
vol = np.array(voxelgrid.get_feature_vector(mode="binary"), dtype=np.uint16)
n_ = 0.01
shape = (n_, n_, n_)
h = shape
mesh = pygalmesh.generate_from_array(vol, [1, 1, 1], cell_size=h[0], facet_distance=h[0], verbose=False)
mesh
# %%
from itertools import chain
points = mesh.points
# faces = [mesh.cells[0].data, mesh.cells[1].data]
faces = list(mesh.cells[0].data) + list(mesh.cells[1].data)
num_face_vertices = np.array([len(face) for face in faces]).reshape(-1, 1)
# pyvista_faces = np.hstack([num_face_vertices, faces]).flatten()
pyvista_faces = list(list(fn) + list(f) for fn, f in zip(num_face_vertices, faces))
# pyvista_faces1 = np.hstack([np.ones([faces1.shape[0], 1])*4, faces1]).flatten()
poly_data_object = pv.PolyData(points, np.array(list(chain(*pyvista_faces))))
poly_data_object.plot(show_edges=True)

# %%
