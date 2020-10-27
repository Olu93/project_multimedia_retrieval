# %%
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
# %%
mesh = pv.read("C:\\Users\\ohund\\workspace\\project_multimedia_retrieval\\trash\\m1.ply")

# mesh.plot()


# %%
def prepare_image(img):
    img_copy = np.ones_like(img)
    img_copy[np.where(img != 0)] = 255
    return img_copy / 255


def extract_sillhouettes(mesh):
    images = []
    normal = np.zeros((3, 1))
    for i in range(3):
        normal[:] = 0
        normal[i] = -1
        cpos = [normal * 2, (0, 0, 0), (0, 0, 1.0)]
        projected = mesh.project_points_to_plane((0, 0, 0), normal=normal)
        _, img = pv.plot(
            projected,
            cpos=cpos,
            notebook=False,
            off_screen=True,
            screenshot=True,
            return_img=True,
            background=[0, 0, 0],
        )
        images.append(prepare_image(img))
    return images


def extract_skeletons(sillh):
    return [binary_closing(skeletonize(img_array, method="zhang")).astype(np.uint8) for img_array in sillh]


def extract_graphs(skeletons):
    graphs = [sknw.build_sknw(ske) for ske in skeletons]
    for G in graphs:
        G.remove_nodes_from(list(nx.isolates(G)))
    return graphs


sillhouettes = extract_sillhouettes(mesh)
plt.imshow(sillhouettes[0], 'gray')
# %%
dims = np.array([1024, 724])
img = proj[0].points
not_null = np.sum(img, axis=0) != 0
points = img[:, not_null]
positive_points = points - points.min(axis=0)
scaled_points = positive_points / positive_points.max(axis=0)
retransformed_points = scaled_points * dims
points = np.floor(retransformed_points).astype(int)
canvas = np.zeros(dims + 1)
canvas[points[:, 0], points[:, 1]] = 1
plt.imshow(canvas.T, cmap="gray")
# %%
fig, ax = plt.subplots()
ax.imshow(canvas, cmap=plt.cm.gray)
contours = skimage.measure.find_contours(canvas, .8)
ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)
# plt.scatter(retransformed_points[:,0], retransformed_points[:,1])
