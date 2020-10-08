# %%
from feature_extractor import FeatureExtractor
from normalizer import Normalizer
from reader import PSBDataset
from helper.config import FEATURE_DATA_FILE, DEBUG, DATA_PATH_NORMED_SUBSET, DATA_PATH_NORMED, CLASS_FILE, DATA_PATH_PSB, DATA_PATH_DEBUG
import time
import pyvista as pv
import pymeshfix as mf
from pyvista import examples
import numpy as np

# %%
cow = examples.download_cow()

# Add holes and cast to triangulated PolyData
cow['random'] = np.random.rand(cow.n_cells)
holy_cow = cow.threshold(0.9, invert=True).extract_geometry().triangulate()
cpos= [(6.56, 8.73, 22.03),
       (0.77, -0.44, 0.0),
       (-0.13, 0.93, -0.35)]

meshfix = mf.MeshFix(holy_cow)
holes = meshfix.extract_holes()

# Render the mesh and outline the holes
p = pv.Plotter()
p.add_mesh(holy_cow, color=True)
p.add_mesh(holes, color='r', line_width=8)
p.camera_position = cpos
p.enable_eye_dome_lighting() # helps depth perception
p.show()

# %%
meshfix = mf.MeshFix(holy_cow)
print(meshfix.extract_holes().n_cells)
meshfix.repair(verbose=True)

repaired = meshfix.mesh
repaired.plot(cpos=cpos)

# %%
