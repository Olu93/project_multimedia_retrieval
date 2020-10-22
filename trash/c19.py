# %%
import numpy as np
from scipy.spatial import ConvexHull
from pyvista import PolyData
from pyvista import examples
import pyvista as pv
# %%
cow = pv.read('..\\processed_data_bkp\\bridge\\m1785.ply')
cow.plot()
# %%
def polyhull(mesh):
    hull = ConvexHull(mesh.points)
    # faces = np.column_stack((3 * np.ones((len(hull.simplices), 1), dtype=np.int), hull.simplices)).flatten()
    poly = PolyData(hull.points)
    poly = poly.delaunay_2d()
    return poly

hull = polyhull(cow)
hull.plot()
# %%
