# %%
import trimesh
from pyvista import examples
import pyvista as pv
import numpy as np

# %%
mesh = examples.download_cow().triangulate()

# %%


def _get_cells(mesh):
    """Returns a list of the cells from this mesh.
    This properly unpacks the VTK cells array.
    There are many ways to do this, but this is
    safe when dealing with mixed cell types."""

    return mesh.faces.reshape(-1, 4)[:, 1:4]


def convert_trimesh2pyvista(trimesh):
    """
    Converts trimesh objects into pyvista objects 
    """
    vertices = trimesh.vertices
    faces = trimesh.faces
    faces_with_number_of_faces = [np.hstack([face.shape[0], face]) for face in faces]
    flattened_faces = np.hstack(faces_with_number_of_faces).flatten()
    return pv.PolyData(vertices, flattened_faces)


def convert_pyvista2trimesh(pvmesh):
    """
    Converts pyvista mesh into trimesh objects
    """
    polygons = [list(p) for p in _get_cells(pvmesh)]
    trimesh_obj = trimesh.Trimesh(vertices=np.array(pvmesh.points), faces=polygons)
    return trimesh_obj


# %%
trimesh_example = convert_pyvista2trimesh(mesh)
trimesh_example.show(show_edges=True)
# %%
pyvista_example = convert_trimesh2pyvista(trimesh_example)
pyvista_example.plot()
# %%
import skeletor as sk
cont = sk.contract(trimesh_example, iter_lim=250)
swc = sk.skeletonize(cont, method='vertex_clusters', sampling_dist=50)
swc = sk.clean(swc, trimesh_example)
swc
# %%
cont.show()
# %%
swc
# %%
new_verts = swc[["x", "y", "z"]]
new_faces = swc[["parent_id", "node_id"]]
skeleton = pv.PolyData(new_verts.values, new_faces.values)
skeleton.plot()
# %%
