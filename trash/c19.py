# %%
import numpy as np
from scipy.spatial import ConvexHull
from pyvista import PolyData
from pyvista import examples
import pyvista as pv
# %%
us_map = pv.read('m1693.ply')
bridge = pv.read('m1785.ply')
# %%
def polyhull(mesh):
    hull = ConvexHull(mesh.points)
    faces = np.column_stack((3 * np.ones((len(hull.simplices), 1), dtype=np.int), hull.simplices)).flatten()
    poly = PolyData(hull.points, faces)
    return poly

hull = polyhull(bridge)
hull.plot()
# %%
def is_flat(mesh):
    return not np.all(mesh.points.sum(axis=0))

print(f"IS FLAT => Bridge: {is_flat(bridge)} | US Map: {is_flat(us_map)}")