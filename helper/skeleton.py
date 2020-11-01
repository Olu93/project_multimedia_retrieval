import itertools
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyvista as pv
import sknw
from skimage.morphology import skeletonize
from skimage.morphology.binary import binary_closing


# pp = pprint.PrettyPrinter(indent=4, stream=logFile)

def exception_catcher_singletons(func):
    def new_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = f"ERR in {func.__name__}: {(str(type(e)), str(e))}"
            print(msg)
            return -1

    return new_func


# def prepare_image(projected):
#     img = projected.points
#     not_null = np.sum(img, axis=0) != 0
#     points = img[:, not_null]

#     positive_points = points - points.min(axis=0)
#     fig = Figure(dpi=20)
#     canvas = FigureCanvasAgg(fig)
#     # print(fig.get_size_inches() * fig.get_dpi())
#     ax = fig.gca()

#     if positive_points.shape[1] > 1:
#         ax.scatter(positive_points[:, 0], positive_points[:, 1], c='k')
#     ax.axis('off')

#     canvas.draw()
#     buf = canvas.buffer_rgba()
#     rgba_image = np.asarray(buf)
#     gray_image = rgba_image.mean(axis=2)
#     normalized = np.ones_like(gray_image)
#     if positive_points.shape[1] > 1:
#         normalized = (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min())
#     image = 1 - normalized
#     image[image != 0] = 1
#     return image


# def extract_sillhouettes(mesh, normal):
#     mesh = mesh.clean()
#     projected = mesh.project_points_to_plane((0, 0, 0), normal=normal)
#     return prepare_image(projected)


def prepare_image(img):
    img_copy = np.ones_like(img)
    img_copy[np.isnan(img)] = 0
    return img_copy.astype(np.uint8)


def extract_sillhouettes(mesh, normal):
    p = pv.Plotter(
        notebook=False,
        off_screen=True,
        window_size=(128, 96),
    )
    projected = mesh.project_points_to_plane((0, 0, 0), normal=normal)
    p.add_mesh(projected)
    p.set_position(normal * 2)
    p.render()
    img = p.get_image_depth()
    p.clear()
    # we need to remove each actor... 
    # https://github.com/pyvista/pyvista/issues/482
    for ren in p.renderers:
        for actor in list(ren._actors):
            ren.remove_actor(actor)
    p.deep_clean()
    del p
    return prepare_image(img)

def extract_graphical_forms(mesh):
    normals = np.eye(3) * -1
    sillhouettes = (extract_sillhouettes(mesh, normal) for normal in normals)
    skeletons = (extract_skeletons(sillh) for sillh in sillhouettes)
    sillh_skel_grph = (extract_graphs(sillh_skel) for sillh_skel in skeletons)
    # time.sleep(1)
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
    if np.isnan(covariance_matrix.sum()):
        return 0
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    size_matters = np.argsort(eigen_values)
    eigen_values, eigen_vectors = eigen_values[size_matters], eigen_vectors[size_matters]
    if eigen_values[1] == 0:
        return 0
    eccentricity = eigen_values[0] / eigen_values[1]
    return eccentricity


@exception_catcher_singletons
def compute_average_shape_thickness(G):
    endpoints = [G.nodes()[node]['o'] for node in extract_endpoints(G)]
    pairs = np.array(list(itertools.product(endpoints, endpoints)))
    differences = np.diff(pairs, axis=1).reshape(-1, 2)
    distances = np.linalg.norm(differences, axis=1)
    return distances.mean()
