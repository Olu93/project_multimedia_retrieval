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
    center_point = np.flip((np.array(skeleton.shape) // 2)).reshape((-1, 2))
    skeleton_points = np.flip(np.array(np.where(skeleton == 1))).T
    # print(center_point.shape)
    # print(skeleton_points.shape)
    return np.linalg.norm(skeleton_points - center_point, axis=1)


compute_distance_to_center(skeletons[0])[:5]
# %%
from skimage.draw import line
import random
skeleton = random.choice(skeletons)
image_shape = np.flip((np.array(skeleton.shape))).reshape(2, -1)
center_point = np.flip((np.array(skeleton.shape) // 2))
skeleton_points = np.flip(np.array(np.where(skeleton == 1)))

# skeleton_points = skeleton_points.T[np.random.choice(len(skeleton_points.T), 100)].T
# skeleton_points = np.array([
#     [600, 600, 600, 600, 600, 600, 600, 550, 620, 640],
#     [700, 750, 690, 650, 620, 670, 590, 650, 650, 650],
# ])
covariance_matrix = np.cov(skeleton_points)
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
canvas = np.zeros_like(skeleton)
size_matters = np.argsort(eigen_values)
eigen_values, eigen_vectors = eigen_values[size_matters], eigen_vectors[size_matters]
print("COV-Matrix: \n", covariance_matrix, "\n")
print("Eigenvector: \n", eigen_vectors, "\n")
print("Eigenvalues: \n", eigen_values, "\n")
eccentiricity = eigen_values[0] / eigen_values[1]
print(eccentiricity)

endpoint_1 = (center_point + (250 * eccentiricity) * eigen_vectors[0]).astype(np.int16)
endpoint_2 = (center_point + (250) * eigen_vectors[1]).astype(np.int16)
print(endpoint_1)
print(endpoint_2)
rr, cc = line(center_point[0], center_point[1], endpoint_1[0], endpoint_1[1])
rr2, cc2 = line(center_point[0], center_point[1], endpoint_2[0], endpoint_2[1])

plt.imshow(skeleton, cmap="gray")
plt.plot(rr, cc, color='green')
plt.plot(rr2, cc2, color='green')
plt.plot(center_point[0], center_point[1], 'r.')
plt.plot(endpoint_1[0], endpoint_1[1], 'r.')
plt.plot(endpoint_2[0], endpoint_2[1], 'r.')
# plt.plot(skeleton_points.T[:, 0], skeleton_points.T[:, 1], 'b.')

# %%
def compute_img_eccentricity(skeleton):
    skeleton_points = np.flip(np.array(np.where(skeleton == 1)))
    covariance_matrix = np.cov(skeleton_points)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    size_matters = np.argsort(eigen_values)
    eigen_values, eigen_vectors = eigen_values[size_matters], eigen_vectors[size_matters]
    eccentiricity = eigen_values[0] / eigen_values[1]
    return eccentiricity

compute_img_eccentricity(skeleton)
# %%
