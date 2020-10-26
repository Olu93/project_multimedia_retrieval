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
# %%
mesh = pv.read("C:\\Users\\ohund\\workspace\\project_multimedia_retrieval\\trash\\m1.ply")
mesh.plot()


def prepare_image(img):
    img_copy = np.ones_like(img)
    img_copy[np.isnan(img)] = 0
    return img_copy


def extract_sillhouettes(mesh):
    images = []
    normal = np.zeros((3, 1))
    p = pv.Plotter()
    for i in range(3):
        normal[:] = 0
        normal[i] = -1
        # cpos = [normal * 2, (0, 0, 0), (0, 0, 1.0)]
        projected = mesh.project_points_to_plane((0, 0, 0), normal=normal)
        p.add_mesh(projected)
        p.set_position(normal * 2)
        images.append(prepare_image(p.get_image_depth()))
    return images


def extract_skeletons(sillh):
    return [binary_closing(skeletonize(img_array, method="zhang")).astype(np.uint8) for img_array in sillh]


def extract_graphs(skeletons):
    graphs = [sknw.build_sknw(ske) for ske in skeletons]
    for G in graphs:
        G.remove_nodes_from(list(nx.isolates(G)))
    return graphs


sillhouettes = extract_sillhouettes(mesh)
skeletons = extract_skeletons(sillhouettes)
graphs = extract_graphs(skeletons)
extracted_information = list(zip(sillhouettes, skeletons, graphs))
# %%p.plot
fig = plt.figure(figsize=(12, 10))
for idx, (img_array, skeleton) in enumerate(zip(sillhouettes, skeletons)):
    ax = fig.add_subplot(2, 3, idx + 1)
    ax.imshow(img_array, cmap=plt.cm.gray)
    ax = fig.add_subplot(2, 3, 3 + idx + 1)
    ax.imshow(skeleton, cmap=plt.cm.gray)
plt.tight_layout()
plt.show()

# %%

img = sillhouettes[0]


# symmetry
def compute_asymmetry(original):
    flip_horizontal = np.flip(original, 0)
    flip_vertical = np.flip(original, 1)
    h_distance = (original - flip_horizontal) * 0.5
    v_distance = (original - flip_vertical) * 0.5
    h_norm = np.sqrt(np.square(h_distance).sum())
    v_norm = np.sqrt(np.square(v_distance).sum())
    return h_norm + v_norm


compute_asymmetry(img)
# %%

# average distance to center_point
def compute_distance_to_center(skeleton):
    center_point = (np.array(skeleton.shape) // 2).reshape((-1, 2))
    skeleton_points = np.array(np.where(skeleton == 1)).T
    print(center_point.shape)
    print(skeleton_points.shape)
    return np.linalg.norm(skeleton_points - center_point, axis=1)


compute_distance_to_center(skeletons[0])
# %%
# cycles: https://stackoverflow.com/questions/15914684/how-can-i-find-cycles-in-a-skeleton-image-with-python-libraries
r = sillhouettes[1]

contours = skimage.measure.find_contours(r, .5)
fig, ax = plt.subplots()
ax.imshow(r, cmap=plt.cm.gray)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
# %%

# %%
