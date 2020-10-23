import pymeshfix as mf
import trimesh.repair as repair
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from pyvista import PolyData
import numpy as np


def exception_catcher(func):
    def new_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"ERR_{func.__name__} for shape {args[0]['meta_data']['name']}")
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


def convex_hull_transformation(mesh):
    poly = mesh
    try:
        hull = ConvexHull(mesh.points)
        poly = PolyData(hull.points).delaunay_2d()
    except QhullError as e:
        print(f"Convex hull operation failed: {str(type(e))} - {str(e)}")
        print("Using fallback!")
    return poly


def __sphericity_test(mesh):
    return mesh.area < 1 and mesh.volume > 1


def sphericity_computation(mesh):
    return (np.power(np.pi, 1 / 3) * np.power(6 * mesh.volume, 2 / 3)) / mesh.area


def compactness_computation(mesh):
    return np.power(mesh.area, 3) / ((36 * np.pi) * np.square(mesh.volume))


def sphericitiy_compuation_2(mesh): # https://sciencing.com/height-prism-8539712.html
    V_sphere = mesh.volume
    radius = np.power((3 * V_sphere * np.pi) / 4, 1 / 3)
    A_sphere = 4 * np.pi * (radius**2)
    A_particle = mesh.area
    return A_sphere / A_particle
