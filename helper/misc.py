import pymeshfix as mf
import trimesh
import trimesh.repair as repair
from matplotlib.colors import LinearSegmentedColormap
import colorsys
import numpy as np
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


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """

    random_colormap = []

    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1), np.random.uniform(low=0.2, high=1), np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high), np.random.uniform(low=low, high=high), np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    return random_colormap


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


def get_feature_type_positions(cols):
    scalar_positions = [idx for idx, header_name in enumerate(cols) if header_name.startswith("scalar_")]
    hist_positions = [idx for idx, header_name in enumerate(cols) if header_name.startswith("hist_")]
    skeleton_positions = [idx for idx, header_name in enumerate(cols) if header_name.startswith("skeleton_")]
    return {
        "scalar": scalar_positions,
        "hist": hist_positions,
        "skeleton": skeleton_positions,
    }