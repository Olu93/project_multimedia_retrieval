from collections import Counter

import jsonlines
from helper import diameter_computer
from helper.misc import compactness_computation, convex_hull_transformation, exception_catcher, sphericity_computation
from helper.skeleton import compute_asymmetry, compute_conjunctions, compute_distance_to_center, compute_edge_lengths, compute_endpoints, compute_img_eccentricity, extract_graphical_forms, extract_sillhouettes
from helper.config import CLASS_FILE, CLASS_FILE_COARSE, DATA_PATH_NORMED, DATA_PATH_NORMED_SUBSET
from reader import PSBDataset
import numpy as np
import random as r
from scipy.spatial import ConvexHull
from pyvista import PolyData
from pyvista import examples
import pyvista as pv
from tqdm import tqdm
import multiprocessing as mp
from feature_extractor import FeatureExtractor
import matplotlib.pyplot as plt


def skeleton_singleton_features(data):
    mesh = data["poly_data"]
    silh_skeleton_graph_set = extract_graphical_forms(mesh)
    return FeatureExtractor.mono_skeleton_features(silh_skeleton_graph_set)


def convex_hull_volume(data):
    mesh = convex_hull_transformation(data["poly_data"])
    convex_hull_volume_result = mesh.volume
    return {"scalar_convex_hull_volume": convex_hull_volume_result}


def rectangularity(data):
    mesh = convex_hull_transformation(data["poly_data"])
    volume = mesh.volume
    min_max_point = np.array(mesh.bounds).reshape((-1, 2))
    differences = np.abs(np.diff(min_max_point, axis=1))
    obb_volume = np.prod(differences)
    rectangularity_result = volume / (obb_volume if obb_volume != 0 else 0.000000000001)
    return {"scalar_rectangularity": rectangularity_result}


def compactness(data):
    mesh = convex_hull_transformation(data["poly_data"])
    compactness_result = compactness_computation(mesh)
    return {"scalar_compactness": compactness_result}


def sphericity(data):
    mesh = convex_hull_transformation(data["poly_data"])
    sphericity_result = sphericity_computation(mesh)
    return {"scalar_sphericity": min(sphericity_result, 1.0)}


def diameter(data):
    mesh = data["poly_data"]
    return {"scalar_diameter": diameter_computer.compute_diameter(mesh)}


def aabb_volume(data):
    mesh = data["poly_data"]
    length_x, length_y, length_z = np.abs(np.diff(np.reshape(mesh.bounds, (3, 2))))
    return {"scalar_aabb_volume": (length_x * length_y * length_z)[0]}


def surface_area(data):
    mesh = data["poly_data"]
    cell_ids = PSBDataset._get_cells(mesh)
    cell_areas = PSBDataset._get_cell_areas(mesh.points, cell_ids)
    return {"scalar_surface_area": sum(cell_areas)}


def eccentricity(data):
    mesh = data["poly_data"]
    A_cov = np.cov(mesh.points.T)
    eigenvalues, _ = np.linalg.eig(A_cov)
    eigenvalues = np.sort(eigenvalues)
    return {"scalar_eccentricity": np.max(eigenvalues) / np.min(eigenvalues) if np.min(eigenvalues) != 0 else eigenvalues[1]}


def cube_root_volume_four_rand_verts(data):
    # https://stackoverflow.com/a/9866530
    def treaeder_volume(points):
        a, b, c, d = points
        return np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6

    mesh = data["poly_data"]
    random_indices = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1, (FeatureExtractor.number_vertices_sampled, 4))
    quad_points = mesh.points[random_indices, :]
    # volumes = np.array([treaeder_volume(points) for points in quad_points])
    A = quad_points[:, 0] - quad_points[:, 3]
    B = np.cross(quad_points[:, 1] - quad_points[:, 3], quad_points[:, 2] - quad_points[:, 3])
    f_volumes = np.abs(np.einsum("ij,ij -> i", A, B)) / 6
    cube_root = f_volumes**(1 / 3)
    histogram = FeatureExtractor.make_bins(cube_root, FeatureExtractor.number_bins)
    del random_indices
    return {"hist_cube_root_volume_four_rand_verts": histogram}


def angle_three_rand_verts(data):
    # This question quite fitted the case (https://bit.ly/3in7MjH)
    mesh = data["poly_data"]
    indices_triplets = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1, (FeatureExtractor.number_vertices_sampled, 3))
    verts_triplets = np.array([mesh.points[triplet] for triplet in indices_triplets])
    o2_1 = verts_triplets[:, 0] - verts_triplets[:, 1]
    o2_3 = verts_triplets[:, 2] - verts_triplets[:, 1]
    dot_products = np.einsum("ij,ij->i", o2_1, o2_3)
    norm_products = np.linalg.norm(o2_1, axis=1) * np.linalg.norm(o2_3, axis=1)
    cosine_angles = dot_products / norm_products
    angle_rads = np.arccos(cosine_angles)
    angles_degs = np.degrees(angle_rads)
    del indices_triplets
    return {"hist_rand_angle_three_verts": FeatureExtractor.make_bins(angles_degs, FeatureExtractor.number_bins)}


def dist_two_rand_verts(data):
    distances = []
    mesh = data["poly_data"]
    indices_tuples = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1, (FeatureExtractor.number_vertices_sampled, 2))
    verts_tuples = [mesh.points[tup] for tup in indices_tuples]
    distances = np.linalg.norm(np.abs(np.diff(np.array(verts_tuples), axis=1)).reshape(-1, 3), axis=1)
    del indices_tuples
    return {"hist_rand_dist_two_verts": FeatureExtractor.make_bins(distances, FeatureExtractor.number_bins)}


def dist_bar_vert(data):
    distances = []
    mesh = data["poly_data"]
    bary_center = mesh.center
    indices = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1, (FeatureExtractor.number_vertices_sampled, 1))
    rand_verts = mesh.points[indices]
    distances = np.linalg.norm(np.abs(rand_verts.reshape(-1, 3) - bary_center), axis=1)
    del indices
    return {"hist_dist_bar_vert": FeatureExtractor.make_bins(distances, FeatureExtractor.number_bins)}


def dist_sqrt_area_rand_triangle(data):
    mesh = data["poly_data"]
    verts_list = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1, (FeatureExtractor.number_vertices_sampled, 3))
    triangle_areas = PSBDataset._get_cell_areas(mesh.points, verts_list)
    sqrt_areas = np.sqrt(triangle_areas)
    del verts_list
    return {"hist_sqrt_area_rand_three_verts": FeatureExtractor.make_bins(sqrt_areas, FeatureExtractor.number_bins)}


def gaussian_curvature(data):
    mesh = data["poly_data"]
    curvatures = mesh.curvature('gaussian')
    return {"hist_gaussian_curvature": FeatureExtractor.make_bins(curvatures, FeatureExtractor.number_bins)}


def mean_curvature(data):
    mesh = data["poly_data"]
    curvatures = mesh.curvature('mean')
    return {"hist_mean_curvature": FeatureExtractor.make_bins(curvatures, FeatureExtractor.number_bins)}


def make_bins(data, n_bins):
    bins = np.linspace(np.min(data), np.max(data), n_bins)
    indices = np.digitize(data, bins)
    count_dict = dict(sorted(Counter(indices).items()))
    count_dict_without_holes = {idx: count_dict[idx] if idx in count_dict.keys() else 0 for idx in range(1, FeatureExtractor.number_bins + 1)}
    result = np.array(list(count_dict_without_holes.values()))
    return result / result.sum()


pipeline = [
    convex_hull_volume,
    rectangularity,
    compactness,
    sphericity,
    diameter,
    aabb_volume,
    surface_area,
    eccentricity,
    cube_root_volume_four_rand_verts,
    angle_three_rand_verts,
    dist_two_rand_verts,
    dist_bar_vert,
    dist_sqrt_area_rand_triangle,
    gaussian_curvature,
    mean_curvature,
    skeleton_singleton_features,
]

if __name__ == "__main__":
    FE = FeatureExtractor(PSBDataset(DATA_PATH_NORMED, class_file_path=CLASS_FILE, class_file_path_coarse=CLASS_FILE_COARSE))
    # print(len(FE.reader.full_data))
    # FE.run_full_pipeline_slow()
    # FE.run_full_pipeline()
    # with mp.Pool(5) as pool:
    # data = [r.choice([0, 1]) for _ in range(2000)]
    params = list(zip(FE.reader.full_data,
                      len(FE.reader.full_data) * [pipeline]))
    results = mp.Pool(9).imap(FE.mono_run_pipeline_debug, tqdm(params))
    with jsonlines.open("computed_features.jsonl", mode="w", flush=True) as writer:
        for item in results:
            writer.write(item)
    # plt.imshow(results[0][0])
    # plt.show()
    # print(len(results))
    # print(img.max())
    # print(img.min())
