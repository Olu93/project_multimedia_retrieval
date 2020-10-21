# %%
import numpy as np
from scipy.spatial import ConvexHull
from pyvista import PolyData
from pyvista import examples
# %%
cow = examples.download_bunny().triangulate()
cow.plot()
# %%
def polyhull(mesh):
    hull = ConvexHull(mesh.points)
    faces = np.column_stack((3 * np.ones((len(hull.simplices), 1), dtype=np.int), hull.simplices)).flatten()
    poly = PolyData(hull.points, faces)
    return poly

hull = polyhull(cow)
hull.plot()
# %%
