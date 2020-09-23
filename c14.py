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
import trimesh

# %%
# mesh = examples.download_cow().triangulate()
# mesh = pv.read("apple.off")
mesh = examples.download_bunny().triangulate()
mesh = pv.PolyData(mesh.points, mesh.cells)
polygons = [list(p) for p in DataSet._get_cells(mesh)]
mesh.plot(color="w", show_edges=True)
# %%
n_faces = mesh.n_faces
deci = mesh.decimate(1-(2500/n_faces))
deci.plot(color="w", show_edges=True)

# %%
subd = deci.subdivide(2)
subd.plot(color="w", show_edges=True)
# %%
clus = pyacvd.Clustering(subd)
# mesh is not dense enough for uniform remeshing
clus.cluster(1000)
remesh = clus.create_mesh()
remesh.plot(color='w', show_edges=True)
# %%
