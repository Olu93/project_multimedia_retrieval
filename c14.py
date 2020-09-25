# %%
import pyacvd
import pyvista as pv

# import pygalmesh
# from pyntcloud import PyntCloud
# import trimesh

# %%
# mesh = examples.download_cow().triangulate()
mesh = pv.read("m1403.off")
# mesh = examples.download_bunny().triangulate()

mesh = pv.PolyData(mesh.points, mesh.cells)
# polygons = [list(p) for p in DataSet._get_cells(mesh)]
mesh.plot(color="w", show_edges=True)
# %%
n_faces = mesh.n_faces
deci = mesh.decimate(1 - (500 / n_faces))
deci.plot(color="w", show_edges=True)

# %%
select = mesh.select_enclosed_points(mesh, check_surface=False)
thresh = .001
inside = select.threshold(thresh)
outside = select.threshold(thresh, invert=True)

p = pv.Plotter()
p.add_mesh(outside, color="Crimson", show_edges=True)
p.add_mesh(inside, color="green", show_edges=True)

p.show()

# %%
# deci.triangulate(inplace=True)
deci.clean(inplace=True)
# deci.smooth(inplace=True)
# %%
subd = mesh.subdivide(3)
subd.plot(color="w", show_edges=True)
# %%
clus = pyacvd.Clustering(deci)
clus.subdivide(3)
# mesh is not dense enough for uniform remeshing
clus.cluster(10000)
remesh = clus.create_mesh()
remesh.plot(color='w', show_edges=True)
# %%
clus.mesh.n_points
