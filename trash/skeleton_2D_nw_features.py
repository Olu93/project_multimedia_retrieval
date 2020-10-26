# %%
from networkx.classes import graph
from networkx.exception import NetworkXNoCycle
import numpy as np
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
import networkx as nx

# %%
mesh = pv.read("C:\\Users\\ohund\\workspace\\project_multimedia_retrieval\\trash\\m1.ply")
mesh.plot()


# %%
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


# %%
def visualize_skeleton_extraction(sillhouettes, skeletons, graphs):
    fig = plt.figure(figsize=(12, 10))
    for idx, (img_array, skeleton, G) in enumerate(zip(sillhouettes, skeletons, graphs)):
        if not len(G.nodes()):
            continue
        ax = fig.add_subplot(3, 3, idx + 1)
        ax.imshow(img_array, cmap=plt.cm.gray)
        ax = fig.add_subplot(3, 3, 3 + idx + 1)
        ax.imshow(skeleton, cmap=plt.cm.gray)
        ax = fig.add_subplot(3, 3, 6 + idx + 1)
        ax.imshow(img_array, cmap="gray")
        for (s, e) in G.edges():
            ps = G[s][e]['pts']
            ax.plot(ps[:, 1], ps[:, 0], 'green')
        nodes = [node for node in G.nodes()]
        ps = np.array([G.nodes()[i]['o'] for i in nodes])
        ax.plot(ps[:, 1], ps[:, 0], 'r.')

    plt.tight_layout()
    return plt.show()


visualize_skeleton_extraction(sillhouettes, skeletons, graphs)


# %%
def extract_endpoints(G):
    return [node for node in G.nodes() if len(G.adj[node]) == 1]


def extract_conjunctions(G):
    return [node for node in G.nodes() if len(G.adj[node]) > 1]


def extract_loops(G):
    try:
        return list(nx.find_cycle(G, orientation="ignore"))
    except NetworkXNoCycle as no_cycle:
        pass


def visualize_skeleton_graph_extraction(sillhouette, G, endpoints=True, conjuncts=True, edge_lengths=True):
    def create_base_image(ax, sillhouette, G):
        ax.imshow(sillhouette, cmap='gray')
        for (s, e) in G.edges():
            ps = G[s][e]['pts']
            ax.plot(ps[:, 1], ps[:, 0], 'green')
        return ax

    num = sum([endpoints, conjuncts, edge_lengths])
    fig, axes = plt.subplots(1, num + 1)
    axes_list = list(axes.ravel()[::-1])
    create_base_image(axes_list.pop(), sillhouette, G)
    if endpoints:
        nodes = extract_endpoints(G)
        ax = create_base_image(axes_list.pop(), sillhouette, G)
        ps = np.array([G.nodes()[i]['o'] for i in nodes])
        ax.plot(ps[:, 1], ps[:, 0], 'r.')
    if conjuncts:
        nodes = extract_endpoints(G)
        ax = create_base_image(axes_list.pop(), sillhouette, G)
        ps = np.array([G.nodes()[i]['o'] for i in nodes])
        ax.plot(ps[:, 1], ps[:, 0], 'r.')


which = 2
visualize_skeleton_graph_extraction(extracted_information[which][0], extracted_information[which][2])
# %%


def extract_edge_lengths(G):
    points = [G[s][e]['pts'] for (s, e) in G.edges()]
    pairings = np.array([p[0] - p[-1] for p in points])
    l2_distances = np.sqrt(np.square(pairings).sum(axis=1))
    return l2_distances



extract_edge_lengths(extracted_information[which][2])
# %%

# %%
