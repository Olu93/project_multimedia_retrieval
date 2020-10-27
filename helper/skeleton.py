import itertools
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyvista as pv
import sknw
from skimage.morphology import skeletonize
from skimage.morphology.binary import binary_closing


def exception_catcher_singletons(func):
    def new_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = f"ERR in {func.__name__}: {(str(type(e)), str(e))}"
            print(msg)
            return -1

    return new_func


def prepare_image_fullimage(img):
    img_copy = img.mean(axis=2)
    img_copy[np.where(img_copy != 0)] = 255
    return img_copy / 255


def prepare_image(img):
    img_copy = np.ones_like(img)
    img_copy[np.isnan(img)] = 0
    return img_copy


def extract_sillhouettes(mesh, normal):
    p = pv.Plotter(
        notebook=False,
        off_screen=True,
    )
    projected = mesh.project_points_to_plane((0, 0, 0), normal=normal)
    p.add_mesh(projected)
    p.set_position(normal * 2)
    p.render()
    img = p.get_image_depth()
    return prepare_image(img)


def extract_sillhouettes_fullimage(mesh, normal):
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
    
    return prepare_image(img)


def extract_graphical_forms(mesh):
    normals = np.eye(3) * -1
    sillhouettes = (extract_sillhouettes(mesh, normal) for normal in normals)
    skeletons = (extract_skeletons(sillh) for sillh in sillhouettes)
    sillh_skel_grph = (extract_graphs(sillh_skel) for sillh_skel in skeletons)
    return list(sillh_skel_grph)


def extract_skeletons(sillh):
    return (sillh, binary_closing(skeletonize(sillh, method="zhang")).astype(np.uint8))


def extract_graphs(sillh_skel):
    G = sknw.build_sknw(sillh_skel[1])
    G.remove_nodes_from(list(nx.isolates(G)))
    return (sillh_skel[0], sillh_skel[1], G)


def extract_endpoints(G):
    return [node for node in G.nodes() if len(G.adj[node]) == 1]


def extract_conjunctions(G):
    return [node for node in G.nodes() if len(G.adj[node]) > 1]


def extract_edge_lengths_fast(G):
    l2_distances = np.array([G[s][e]['weight'] for s, e in G.edges()])
    return l2_distances if len(l2_distances) else np.array([0])


@exception_catcher_singletons
def compute_endpoints(G):
    return len(extract_endpoints(G))


@exception_catcher_singletons
def compute_conjunctions(G):
    return len(extract_conjunctions(G))


@exception_catcher_singletons
def compute_edge_lengths(G):
    return extract_edge_lengths_fast(G).mean()


@exception_catcher_singletons
def compute_asymmetry(original):
    flip_horizontal = np.flip(original, 0)
    flip_vertical = np.flip(original, 1)
    h_distance = (original - flip_horizontal) * 0.5
    v_distance = (original - flip_vertical) * 0.5
    h_norm = np.sqrt(np.square(h_distance).sum())
    v_norm = np.sqrt(np.square(v_distance).sum())
    return float(h_norm + v_norm)


@exception_catcher_singletons
def compute_distance_to_center(skeleton):
    center_point = np.flip((np.array(skeleton.shape) // 2)).reshape((-1, 2))
    skeleton_points = np.flip(np.array(np.where(skeleton == 1))).T  # TODO: Use node points instead
    if not len(skeleton_points):
        return 0
    return np.linalg.norm(skeleton_points - center_point, axis=1).mean()


@exception_catcher_singletons
def compute_img_eccentricity(skeleton):
    skeleton_points = np.array(np.where(skeleton == 1))
    if skeleton_points.size == 0:
        return 0
    skeleton_points = np.flip(skeleton_points)
    covariance_matrix = np.cov(skeleton_points)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    size_matters = np.argsort(eigen_values)
    eigen_values, eigen_vectors = eigen_values[size_matters], eigen_vectors[size_matters]
    eccentricity = eigen_values[0] / eigen_values[1]
    return eccentricity


@exception_catcher_singletons
def compute_average_shape_thickness(G):
    endpoints = [G.nodes()[node]['o'] for node in extract_endpoints(G)]
    pairs = np.array(list(itertools.product(endpoints, endpoints)))
    differences = np.diff(pairs, axis=1).reshape(-1, 2)
    distances = np.linalg.norm(differences, axis=1)
    return distances.mean()
