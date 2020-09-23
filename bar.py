#%%
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

#  %%

# cow = cow.subdivide(2)
# cow = cow.decimate_pro(.7)
# cow = examples.download_cow().triangulate()
cow = pv.read("apple.off")
# polygons = DataSet._get_cells(cow)
geom = pygmsh.built_in.Geometry()

# %%
poly = geom.add_polygon(cow.points)

mesh = pygmsh.generate_mesh(geom)

# %%
