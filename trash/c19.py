# %%
from matplotlib import pyplot
import numpy as np
from scipy.spatial import ConvexHull
from pyvista import PolyData
from pyvista import examples
import pyvista as pv
# %%
us_map = pv.read('m1693.ply')
bridge = pv.read('m1785.ply')
watch = pv.read('m598.ply')
# %%
def polyhull(mesh):
    hull = ConvexHull(mesh.points)
    faces = np.column_stack((3 * np.ones((len(hull.simplices), 1), dtype=np.int), hull.simplices)).flatten()
    poly = PolyData(hull.points, faces)
    return poly

hull = polyhull(watch)
# hull.plot()
# %%
def is_flat(mesh):
    return not np.all(mesh.points.sum(axis=0))

print(f"IS FLAT => Bridge: {is_flat(bridge)} | US Map: {is_flat(us_map)}")

# %%
plotter = pv.Plotter(shape=(1, 2), notebook=False)
plotter.subplot(0, 0)
# plotter.add_text("A watch", font_size=30)
plotter.add_mesh(watch)

plotter.subplot(0, 1)
# plotter.add_text("Convex Hull", font_size=30)
plotter.add_mesh(hull)
plotter.link_views()
# plotter.show()
# plotter
# plotter.screenshot('watch_convex_hull.png', transparent_background=True)
camera = plotter.show(screenshot='watch_convex_hull.png')


plotter = pv.Plotter(shape=(1, 2), notebook=False)
plotter.subplot(0, 0)
# plotter.add_text("A watch", font_size=30)
plotter.add_mesh(watch)
plotter.camera_position = camera

plotter.subplot(0, 1)
# plotter.add_text("Convex Hull", font_size=30)
plotter.add_mesh(hull)
plotter.link_views()
# plotter.show()
# plotter
plotter.screenshot('watch_convex_hull.png', transparent_background=True)
# %%
