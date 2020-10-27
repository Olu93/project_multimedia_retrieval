import pymeshfix as mf
import trimesh
import trimesh.repair as repair
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from pyvista import PolyData
import numpy as np
import pyvista as pv


def exception_catcher(func):
    def new_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"ERR_{func.__name__} for shape {args[0]['meta_data']['name']}: {(str(type(e)), str(e))}")
            return {f"ERR_{func.__name__}": (str(type(e)), str(e))}

    return new_func


def fill_holes_old(mesh):
    # https://pymeshfix.pyvista.org/index.html: M. Attene. A lightweight approach to repairing digitized polygon meshes. The Visual Computer, 2010. (c) Springer. DOI: 10.1007/s00371-010-0416-3
    meshfix = mf.MeshFix(mesh)
    meshfix.repair(verbose=True)
    repaired = meshfix.mesh
    return repaired


def fill_holes(mesh):
    return mesh.fill_holes(1000)


def is_flat(mesh):
    return not np.all(mesh.points.sum(axis=0))


def convex_hull_transformation(mesh):
    poly = mesh
    if is_flat(poly):
        return poly
    try:
        hull = ConvexHull(mesh.points)
        faces = np.column_stack((3 * np.ones((len(hull.simplices), 1), dtype=np.int), hull.simplices)).flatten()
        poly = PolyData(hull.points, faces)
    except QhullError as e:
        print(f"Convex hull operation failed: {str(type(e))} - {str(e)}")
        print("Using fallback!")
    return poly


def __sphericity_test(mesh):
    return mesh.area < 1 and mesh.volume > 1


def sphericity_computation(mesh):
    if mesh.area == 0:
        return 0
    return (np.power(np.pi, 1 / 3) * np.power(6 * mesh.volume, 2 / 3)) / mesh.area


def compactness_computation(mesh):
    if mesh.volume == 0:
        return 0
    return np.power(mesh.area, 3) / ((36 * np.pi) * np.square(mesh.volume))


def sphericitiy_compuation_2(mesh):  # https://sciencing.com/height-prism-8539712.html
    V_sphere = mesh.volume
    radius = np.power((3 * V_sphere * np.pi) / 4, 1 / 3)
    A_sphere = 4 * np.pi * (radius**2)
    A_particle = mesh.area
    return A_sphere / A_particle


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


def jsonify(item):
    result = {key: list(value) if type(value) in [np.ndarray] else value for key, value in item.items()}
    return result