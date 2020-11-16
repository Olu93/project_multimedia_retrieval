# %%
from PIL.Image import Image
import numpy as np
from numpy.core.defchararray import center
from scipy.spatial import ConvexHull
from pyvista import PolyData
from pyvista import examples
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.morphology.binary import binary_closing
from skimage.util import dtype
from skimage.morphology import skeletonize, binary_dilation, binary_erosion, skeletonize_3d
from skimage.util import invert
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.spatial.qhull import QhullError
import skimage
import sknw
import cv2
import networkx as nx
import pandas as pd
import sys
from descartes import PolygonPatch
import matplotlib.pyplot as plt
import alphashape
from scipy.interpolate import griddata
from polylidar import extractPlanesAndPolygons, extractPolygons, Delaunator
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
# %%
mesh = pv.read("C:\\Users\\ohund\\workspace\\project_multimedia_retrieval\\trash\\m1.ply")

# mesh.plot()


# %%
def prepare_image(img, proj):
    img_copy = np.ones_like(img)
    img_copy[np.isnan(img)] = 0
    return img_copy, proj


def extract_sillhouettes(mesh):
    images = []
    normal = np.zeros((3, 1))
    p = pv.Plotter(
        notebook=False,
        off_screen=True,
    )
    for i in range(3):
        normal[:] = 0
        normal[i] = -1
        projected = mesh.project_points_to_plane((0, 0, 0), normal=normal)
        p.add_mesh(projected)
        p.set_position(normal * 2)
        img = p.get_image_depth()
        images.append(prepare_image(img, projected))
    return images


def extract_skeletons(sillh):
    return [binary_closing(skeletonize(img_array, method="zhang")).astype(np.uint8) for img_array in sillh]


def extract_graphs(skeletons):
    graphs = [sknw.build_sknw(ske) for ske in skeletons]
    for G in graphs:
        G.remove_nodes_from(list(nx.isolates(G)))
    return graphs


sillhouettes = extract_sillhouettes(mesh)
plt.imshow(sillhouettes[2][0], 'gray')
# %%
projected = sillhouettes[2][1]
dims = np.array([1024, 724])
img = projected.points
not_null = np.sum(img, axis=0) != 0
points = img[:, not_null]


# %%
import time
import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from polylidar import extractPlanesAndPolygons
from polylidarutil import (generate_test_points, plot_points, plot_triangles, get_estimated_lmax,
                            plot_triangle_meshes, get_triangles_from_he, get_plane_triangles, plot_polygons)


kwargs = dict(num_groups=2, group_size=1000, dist=100.0, seed=1)
# generate random normally distributed clusters of points, 200 X 2 numpy array.
new_points = points
# Extracts planes and polygons, time
t1 = time.time()
delaunay, planes, polygons = extractPlanesAndPolygons(new_points, alpha=1, lmax=1)
t2 = time.time()
print("Took {:.2f} milliseconds".format((t2 - t1) * 1000))

# Plot Data
if new_points.shape[0] < 100000:
    fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
    # plot points
    plot_points(new_points, ax)
    # plot all triangles
    # plot_triangles(get_triangles_from_he(delaunay.triangles, points), ax)
    # plot mesh triangles
    # triangle_meshes = get_plane_triangles(planes, delaunay.triangles, points)
    # plot_triangle_meshes(triangle_meshes, ax)
    # plot polygons
    # plot_polygons(polygons, delaunay, new_points, ax)

    plt.axis('equal')

    plt.show()