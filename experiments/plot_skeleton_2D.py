from helper.misc import screenshot_mesh
import io
import pathlib
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
import itertools
import json

def prepare_image(img):
    img_copy = np.ones_like(img)
    img_copy[np.isnan(img)] = 0
    return img_copy.astype(np.uint8)


def extract_sillhouettes(mesh, normal):
    p = pv.Plotter(
        notebook=False,
        off_screen=True,
        # window_size=(432, 288),
    )
    projected = mesh.project_points_to_plane((0, 0, 0), normal=normal)
    p.add_mesh(projected)
    p.set_position(normal * 2)
    p.render()
    img = p.get_image_depth()

    return prepare_image(img)


def extract_skeletons(sillh):
    return [binary_closing(skeletonize(img_array, method="zhang")).astype(np.uint8) for img_array in sillh]


def extract_graphs(skeletons):
    graphs = [sknw.build_sknw(ske) for ske in skeletons]
    for G in graphs:
        G.remove_nodes_from(list(nx.isolates(G)))
    return graphs


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
    fig, axes = plt.subplots(1, num)
    axes_list = list(axes.ravel()[::-1])
    create_base_image(axes_list.pop(), sillhouette, G)
    if endpoints:
        nodes = extract_endpoints(G)
        ax = create_base_image(axes_list.pop(), sillhouette, G)
        ps = np.array([G.nodes()[i]['o'] for i in nodes])
        ax.plot(ps[:, 1], ps[:, 0], 'r.')
    if conjuncts:
        nodes = extract_conjunctions(G)
        ax = create_base_image(axes_list.pop(), sillhouette, G)
        ps = np.array([G.nodes()[i]['o'] for i in nodes])
        ax.plot(ps[:, 1], ps[:, 0], 'r.')


def extract_edge_lengths(G):
    points = [G[s][e]['pts'] for (s, e) in G.edges()]
    pairings = np.array([p[0] - p[-1] for p in points])
    l2_distances = np.sqrt(np.square(pairings).sum(axis=1))
    return l2_distances


# print(extract_edge_lengths(extracted_information[which][2])[:5])


def extract_edge_lengths_fast(G):
    l2_distances = np.array([G[s][e]['weight'] for s, e in G.edges()])
    return l2_distances


# print(extract_edge_lengths_fast(extracted_information[which][2])[:5])


def extract_edge_paths(G):
    def follow_paths(path_start):
        if type(path_start) == list:
            return path_start
        path_set = []
        for k, v in path_start.items():
            path_set.extend(follow_paths(v))
        return tuple(set(path_set))

    H = G.__class__()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges)

    populated_lengths = [node for node in H.nodes() if len(H.adj[node]) == 2]
    H = H.subgraph(populated_lengths)
    complexetons = list(set([follow_paths(v) for k, v in nx.shortest_path(H).items()]))
    complexetons = [keep for keep in complexetons if len(keep) > 1]

    # final_paths =  G.edges(complexetons)

    others = [node for node in G.nodes() if len(G.adj[node]) > 2]
    simpletons = list(set(G.edges(others)))
    return complexetons, simpletons


# grph = extracted_information[which][2]

# extract_edge_paths(grph)


def compute_graph_path_lengths(G):
    complexetons, simpletons = extract_edge_paths(G)
    return [len(pth) + 2 for pth in complexetons] + [1 for pth in simpletons]


# compute_graph_path_lengths(grph)


def compute_average_shape_thickness(G):
    endpoints = [G.nodes()[node]['o'] for node in extract_endpoints(G)]
    pairs = np.array(list(itertools.product(endpoints, endpoints)))
    differences = np.diff(pairs, axis=1).reshape(-1, 2)
    distances = np.linalg.norm(differences, axis=1)
    return distances.mean()


# print(compute_average_shape_thickness(grph))


def visualize_skeleton_extraction(original_image, sillhouettes, skeletons, graphs):

    fig = plt.figure(figsize=(10, 3), facecolor='black')

    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(original_image, cmap=plt.cm.gray)
    ax.axis('off')
    ax.set_title(f"Original image", loc="center", color="white", y=.0)

    for idx, (img_array, skeleton, G, axtitle) in enumerate(zip(sillhouettes, skeletons, graphs, "x y z".split())):
        ax = fig.add_subplot(1, 4, idx + 2)
        ax.imshow(img_array, cmap=plt.cm.gray)
        for (s, e) in G.edges():
            ps = G[s][e]['pts']
            ax.plot(ps[:, 1], ps[:, 0], 'green')
        nodes = [node for node in G.nodes()]
        ps = np.array([G.nodes()[i]['o'] for i in nodes])
        ax.plot(ps[:, 1], ps[:, 0], 'r.')
        ax.set_title(f"Orientation {axtitle}", loc="center", color="white", y=.0)

    # fig.tight_layout()
    fig.suptitle("2D Contour, Skeleton and Graph of a Mesh", color="white", y=0.80)


    return fig


if __name__ == "__main__":
    mesh = pv.read("C:\\Users\\ohund\\workspace\\project_multimedia_retrieval\\trash\\m118.ply")
    cam_pos_file = "trash/skeleton_camera_pos.json"
    if not pathlib.Path(cam_pos_file).exists():
        cam = mesh.plot()
        json.dump(tuple(cam), io.open(cam_pos_file, "w"))
    else:
        cam = json.load(io.open(cam_pos_file))
    orig_image = screenshot_mesh(mesh, cam)
    # orig_image[np.isnan(orig_image)] = 255
    sillhouettes = [extract_sillhouettes(mesh, normal) for normal in np.eye(3) * -1]
    skeletons = extract_skeletons(sillhouettes)
    graphs = extract_graphs(skeletons)
    fig = visualize_skeleton_extraction(orig_image, sillhouettes, skeletons, graphs)
    plt.tight_layout()
    plt.savefig("figs/fig_skeleton.png")
    # plt.show()
    plt.close()