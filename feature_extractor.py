from collections import Counter
from datetime import datetime
from helper.skeleton import compute_conjunctions, compute_distance_to_center, compute_edge_lengths, compute_endpoints, compute_img_eccentricity, extract_graphical_forms, compute_asymmetry
from pprint import pprint
import glob

import jsonlines
import numpy as np
from tqdm import tqdm

from helper import diameter_computer
from helper.misc import compactness_computation, convex_hull_transformation, exception_catcher, fill_holes, jsonify, sphericity_computation
from helper.diameter_computer import compute_diameter
from helper.config import DATA_PATH_NORMED, DEBUG, DATA_PATH_NORMED_SUBSET, CLASS_FILE
from helper.mp_functions import compute_feature_extraction, compute_feature_extraction_old
from reader import PSBDataset
import jsonlines
import io
from os import path
from datetime import datetime
import pyvista as pv
import multiprocessing as mp
import tracemalloc
import faulthandler
# faulthandler.enable() 
# tracemalloc.start()
# TODO: [x] surface area
# TODO: [x] compactness (with respect to a sphere)
# TODO: [x] axis-aligned bounding-box volume
# TODO: [x] diameter
# TODO: [x] eccentricity
# TODO: [x] A3: angle between 3 random vertices
# TODO: [x] D1: distance between barycenter and random vertex
# TODO: [x] D2: distance between 2 random vertices
# TODO: [x] D3: square root of area of triangle given by 3 random vertices
# TODO: [x] D4: cube root of volume of tetrahedron formed by 4 random vertices
# TODO: [ ] Mention m94, m778 removal and m1693 eccentricity stabilisation
# TODO: [ ] Change fill_holes with convex hull operation
#
# np.seterr('raise')


class FeatureExtractor:
    number_vertices_sampled = 100 if DEBUG else 100000
    number_bins = 20

    def __init__(self, reader=None, target_file="./computed_features.jsonl", append_mode=False):
        self.timestamp = str(datetime.now())
        self.feature_stats_file = target_file
        self.append_mode = append_mode
        if reader:
            self.reader = reader
            self.full_data = reader.run_full_pipeline()

    @staticmethod
    def mono_run_pipeline(data):
        final_dict = {}
        final_dict["name"] = data["meta_data"]["name"]
        final_dict["label"] = data["meta_data"]["label"]
        data["poly_data"] = pv.PolyData(data["data"]["vertices"], data["data"]["faces"])
        singleton_pipeline, histogram_pipeline = FeatureExtractor.get_pipeline_functions()

        gather_data = [list(func(data).items())[0] for func in [*list(singleton_pipeline.keys()), *list(histogram_pipeline.keys())]]

        final_dict.update(gather_data)
        # final_dict.update(skeleton_features)
        return final_dict

    @staticmethod
    def mono_run_pipeline_old(data):
        final_dict = {}
        final_dict["name"] = data["meta_data"]["name"]
        final_dict["label"] = data["meta_data"]["label"]
        data["poly_data"] = pv.PolyData(data["data"]["vertices"], data["data"]["faces"])
        singleton_pipeline, histogram_pipeline = FeatureExtractor.get_pipeline_functions()

        gather_data = [list(func(data).items())[0] for func in [*list(singleton_pipeline.keys()), *list(histogram_pipeline.keys())]]

        skeleton_features = FeatureExtractor.skeleton_singleton_features(data)

        final_dict.update(gather_data)
        final_dict.update(skeleton_features)
        return final_dict

    @staticmethod
    def mono_run_pipeline_slow(data):
        final_dict = {}
        final_dict["name"] = data["meta_data"]["name"]
        final_dict["label"] = data["meta_data"]["label"]
        data["poly_data"] = pv.PolyData(data["data"]["vertices"], data["data"]["faces"])
        singleton_pipeline, histogram_pipeline = FeatureExtractor.get_pipeline_functions()
        # histogram_pipeline.update({FeatureExtractor.gaussian_curvature: "Gaussian Curvature"})

        gather_data = [list(func(data).items())[0] for func in [*list(singleton_pipeline.keys()), *list(histogram_pipeline.keys())]]
        skeleton_features = FeatureExtractor.mono_skeleton_features(data["images"])

        final_dict.update(gather_data)
        final_dict.update(skeleton_features)
        return final_dict

    def run_full_pipeline(self, max_num_items=None):
        num_full_data = len(self.full_data)
        relevant_subset_of_data = self.full_data[:min(max_num_items, num_full_data)] if max_num_items else self.reader.full_data
        result = compute_feature_extraction(self, relevant_subset_of_data)

        filenames = glob.glob("stats/tmp/tmp-*.jsonl")
        with open(self.feature_stats_file, 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    outfile.write(infile.read())
        return result

    def run_full_pipeline_old(self, max_num_items=None):

        target_file = self.feature_stats_file
        features = []
        with jsonlines.open(target_file, mode="a" if self.append_mode else "w") as writer:
            num_full_data = len(self.reader.full_data)
            relevant_subset_of_data = self.reader.full_data[:min(max_num_items, num_full_data)] if max_num_items else self.reader.full_data
            num_data_being_processed = len(relevant_subset_of_data)
            feature_data_generator = compute_feature_extraction_old(self, tqdm(relevant_subset_of_data, total=num_data_being_processed))
            prepared_data = (jsonify(item) for item in feature_data_generator)
            prepared_data = (dict(timestamp=self.timestamp, **item) for item in prepared_data)
            for next_feature_set in prepared_data:
                features.append(next_feature_set)
                writer.write(next_feature_set)
        return features

    def run_full_pipeline_slow(self, max_num_items=None):
        self.full_data = self.reader.load_image_data()
        target_file = self.feature_stats_file
        features = []
        with jsonlines.open(target_file, mode="a" if self.append_mode else "w", flush=True) as writer:
            num_full_data = len(self.reader.full_data)
            relevant_subset_of_data = self.reader.full_data[:min(max_num_items, num_full_data)] if max_num_items else self.reader.full_data
            num_data_being_processed = len(relevant_subset_of_data)
            prepared_data = (FeatureExtractor.mono_run_pipeline_slow(data) for data in tqdm(relevant_subset_of_data, total=num_data_being_processed))
            prepared_data = (jsonify(item) for item in prepared_data)
            prepared_data = (dict(timestamp=self.timestamp, **item) for item in prepared_data)
            for next_feature_set in prepared_data:
                features.append(next_feature_set)
                writer.write(next_feature_set)
        return features

    @staticmethod
    def get_pipeline_functions():
        singleton_pipeline = {
            FeatureExtractor.compactness: "Compactness",
            FeatureExtractor.sphericity: "Sphericity",
            FeatureExtractor.aabb_volume: "Bounding Box Volume",
            FeatureExtractor.surface_area: "Surface Area",
            FeatureExtractor.eccentricity: "Eccentricity",
            FeatureExtractor.diameter: "Diameter",
            FeatureExtractor.convex_hull_volume: "Convex Hull Volume",
            FeatureExtractor.rectangularity: "Rectangularity",
        }
        # if not DEBUG:
        #     singleton_pipeline[FeatureExtractor.diameter] = "Diameter"
        histogram_pipeline = {
            FeatureExtractor.angle_three_rand_verts: "Angl. of sampled vert. triplets",
            FeatureExtractor.dist_bar_vert: "Dist. between sampled vert. & barycenter",
            FeatureExtractor.dist_two_rand_verts: "Dist. of sampled vert. pairs",
            FeatureExtractor.dist_sqrt_area_rand_triangle: "Sqrt. of sampled triangles",
            FeatureExtractor.cube_root_volume_four_rand_verts: "Curt. of sampled tetrahedrons",
            # FeatureExtractor.gaussian_curvature: "Gaussian Curvature",
            # FeatureExtractor.mean_curvature: "Mean Curvature",
        }
        return (singleton_pipeline, histogram_pipeline)

    @staticmethod
    @exception_catcher
    def skeleton_singleton_features(data):
        mesh = data["poly_data"]
        silh_skeleton_graph_set = extract_graphical_forms(mesh)
        return FeatureExtractor.mono_skeleton_features(silh_skeleton_graph_set)

    @staticmethod
    @exception_catcher
    def mono_skeleton_features(silh_skeleton_graph_set):
        num_endpoints = [compute_endpoints(grph) for silh, skel, grph in silh_skeleton_graph_set]
        num_conjunctions = [compute_conjunctions(grph) for silh, skel, grph in silh_skeleton_graph_set]
        asymmetries = [compute_asymmetry(silh) for silh, skel, grph in silh_skeleton_graph_set]
        img_eccentricity = [compute_img_eccentricity(skel) for silh, skel, grph in silh_skeleton_graph_set]
        average_edge_lengths = [compute_edge_lengths(grph) for silh, skel, grph in silh_skeleton_graph_set]
        average_distance_to_center = [compute_distance_to_center(skel) for silh, skel, grph in silh_skeleton_graph_set]
        del silh_skeleton_graph_set
        return {
            "skeleton_num_endpoints": num_endpoints,
            "skeleton_num_conjunctions": num_conjunctions,
            "skeleton_asymmetries": asymmetries,
            "skeleton_avg_distances": average_distance_to_center,
            "skeleton_avg_edgelength": average_edge_lengths,
            "skeleton_img_eccentricity": img_eccentricity,
        }

    @staticmethod
    @exception_catcher
    def convex_hull_volume(data):
        mesh = convex_hull_transformation(data["poly_data"])
        convex_hull_volume_result = mesh.volume
        return {"convex_hull_volume": convex_hull_volume_result}

    @staticmethod
    @exception_catcher
    def rectangularity(data):
        mesh = convex_hull_transformation(data["poly_data"])
        volume = mesh.volume
        min_max_point = np.array(mesh.bounds).reshape((-1, 2))
        differences = np.abs(np.diff(min_max_point, axis=1))
        obb_volume = np.prod(differences)
        rectangularity_result = volume / (obb_volume if obb_volume != 0 else 0.000000000001)
        return {"rectangularity": rectangularity_result}

    @staticmethod
    @exception_catcher
    def compactness(data):
        mesh = convex_hull_transformation(data["poly_data"])
        compactness_result = compactness_computation(mesh)
        return {"compactness": compactness_result}

    @staticmethod
    @exception_catcher
    def sphericity(data):
        mesh = convex_hull_transformation(data["poly_data"])
        sphericity_result = sphericity_computation(mesh)
        return {"sphericity": min(sphericity_result, 1.0)}

    @staticmethod
    @exception_catcher
    def diameter(data):
        mesh = data["poly_data"]
        return {"diameter": diameter_computer.compute_diameter(mesh)}

    @staticmethod
    @exception_catcher
    def aabb_volume(data):
        mesh = data["poly_data"]
        length_x, length_y, length_z = np.abs(np.diff(np.reshape(mesh.bounds, (3, 2))))
        return {"aabb_volume": (length_x * length_y * length_z)[0]}

    @staticmethod
    @exception_catcher
    def surface_area(data):
        mesh = data["poly_data"]
        cell_ids = PSBDataset._get_cells(mesh)
        cell_areas = PSBDataset._get_cell_areas(mesh.points, cell_ids)
        return {"surface_area": sum(cell_areas)}

    @staticmethod
    @exception_catcher
    def eccentricity(data):
        mesh = data["poly_data"]
        A_cov = np.cov(mesh.points.T)
        eigenvalues, _ = np.linalg.eig(A_cov)
        eigenvalues = np.sort(eigenvalues)
        return {"eccentricity": np.max(eigenvalues) / np.min(eigenvalues) if np.min(eigenvalues) != 0 else eigenvalues[1]}

    @staticmethod
    @exception_catcher
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
        return {"cube_root_volume_four_rand_verts": histogram}

    @staticmethod
    @exception_catcher
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
        return {"rand_angle_three_verts": FeatureExtractor.make_bins(angles_degs, FeatureExtractor.number_bins)}

    @staticmethod
    @exception_catcher
    def dist_two_rand_verts(data):
        distances = []
        mesh = data["poly_data"]
        indices_tuples = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1, (FeatureExtractor.number_vertices_sampled, 2))
        verts_tuples = [mesh.points[tup] for tup in indices_tuples]
        distances = np.linalg.norm(np.abs(np.diff(np.array(verts_tuples), axis=1)).reshape(-1, 3), axis=1)
        del indices_tuples
        return {"rand_dist_two_verts": FeatureExtractor.make_bins(distances, FeatureExtractor.number_bins)}

    @staticmethod
    @exception_catcher
    def dist_bar_vert(data):
        distances = []
        mesh = data["poly_data"]
        bary_center = mesh.center
        indices = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1, (FeatureExtractor.number_vertices_sampled, 1))
        rand_verts = mesh.points[indices]
        distances = np.linalg.norm(np.abs(rand_verts.reshape(-1, 3) - bary_center), axis=1)
        del indices
        return {"dist_bar_vert": FeatureExtractor.make_bins(distances, FeatureExtractor.number_bins)}

    @staticmethod
    @exception_catcher
    def dist_sqrt_area_rand_triangle(data):
        mesh = data["poly_data"]
        verts_list = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1, (FeatureExtractor.number_vertices_sampled, 3))
        triangle_areas = PSBDataset._get_cell_areas(mesh.points, verts_list)
        sqrt_areas = np.sqrt(triangle_areas)
        del verts_list
        return {"sqrt_area_rand_three_verts": FeatureExtractor.make_bins(sqrt_areas, FeatureExtractor.number_bins)}

    @staticmethod
    @exception_catcher
    def gaussian_curvature(data):
        mesh = data["poly_data"]
        curvatures = mesh.curvature('gaussian')
        return {"gaussian_curvature": FeatureExtractor.make_bins(curvatures, FeatureExtractor.number_bins)}

    @staticmethod
    @exception_catcher
    def mean_curvature(data):
        mesh = data["poly_data"]
        curvatures = mesh.curvature('mean')
        return {"mean_curvature": FeatureExtractor.make_bins(curvatures, FeatureExtractor.number_bins)}

    @staticmethod
    def make_bins(data, n_bins):
        bins = np.linspace(np.min(data), np.max(data), n_bins)
        indices = np.digitize(data, bins)
        count_dict = dict(sorted(Counter(indices).items()))
        count_dict_without_holes = {idx: count_dict[idx] if idx in count_dict.keys() else 0 for idx in range(1, FeatureExtractor.number_bins + 1)}
        result = np.array(list(count_dict_without_holes.values()))
        return result / result.sum()

    @staticmethod
    def generate_random_ints(min_val, max_val, shape):
        return np.array([np.random.choice(line, shape[1], replace=False) for line in np.repeat(np.arange(min_val, max_val), 10000, axis=0).reshape(max_val, -1).T])

    @staticmethod
    def generate_random_ints_(min_val, max_val, shape):
        def sorting(a):
            b = np.sort(a, axis=1)
            return (b[:, 1:] != b[:, :-1]).sum(axis=1) + 1

        X = np.arange(0, max_val)
        X_samples = np.random.choice(X, shape[0], replace=False)
        X_combinations_raw = np.array(np.meshgrid(*[X_samples] * shape[1])).T.reshape(-1, shape[1])
        X_unique_counts = sorting(X_combinations_raw)
        X_combinations_unique = X_combinations_raw[X_unique_counts == shape[1]]
        return X_combinations_unique


class TsneVisualiser:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        pass


if __name__ == "__main__":
    FE = FeatureExtractor(PSBDataset(DATA_PATH_NORMED_SUBSET if DEBUG else DATA_PATH_NORMED, class_file_path=CLASS_FILE))
    # FE.run_full_pipeline_slow()
    FE.run_full_pipeline()
    # FE.run_full_pipeline_old()
