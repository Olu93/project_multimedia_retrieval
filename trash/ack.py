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
mesh = examples.download_cow().triangulate()
polygons = [list(p) for p in DataSet._get_cells(mesh)]
# %%
trimesh_obj = trimesh.Trimesh(vertices=np.array(mesh.points), faces=polygons)
trimesh_obj.show(show_edges=True)
# %%
trimesh.remesh