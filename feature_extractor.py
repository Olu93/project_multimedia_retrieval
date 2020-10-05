from collections import Counter
from datetime import datetime
from pprint import pprint

import jsonlines
import numpy as np
from tqdm import tqdm

from helper import diameter_computer
from helper.config import DEBUG, DATA_PATH_NORMED_SUBSET
from helper.misc import exception_catcher
from helper.diameter_computer import compute_diameter
from helper.config import DATA_PATH_NORMED, DEBUG, DATA_PATH_NORMED_SUBSET
from helper.mp_functions import compute_feature_extraction
from reader import PSBDataset
import jsonlines
import io
from os import path
from datetime import datetime
import pyvista as pv
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


class FeatureExtractor:
    number_vertices_sampled = 1000
    number_bins = 20

    def __init__(self, data_path=None, target_file="./computed_features.jsonl", append_mode=False):
        assert data_path, "Plzzz provide a data path. Wherez all the data???"
        self.reader = PSBDataset(search_path=data_path)
        self.reader.read()
        self.reader.load_files_in_memory()
        self.reader.convert_all_to_polydata()
        self.reader.compute_shape_statistics()
        self.full_data = self.reader.full_data
        self.timestamp = str(datetime.now())
        self.feature_stats_file = target_file
        self.append_mode = append_mode

    @staticmethod
    def mono_run_pipeline(data):
        final_dict = {}
        final_dict["name"] = data["meta_data"]["name"]
        data["poly_data"] = pv.PolyData(data["data"]["vertices"], data["data"]["faces"])
        singleton_pipeline = [
            FeatureExtractor.compactness,
            FeatureExtractor.sphericity,
            FeatureExtractor.aabb_volume,
            FeatureExtractor.surface_area,
            FeatureExtractor.eccentricity,
        ]
        histogram_pipeline = [
            FeatureExtractor.angle_three_rand_verts,
            FeatureExtractor.dist_bar_vert,
            FeatureExtractor.dist_two_rand_verts,
            FeatureExtractor.dist_sqrt_area_rand_triangle,
            FeatureExtractor.cube_root_volume_four_rand_verts,
        ]
        if not DEBUG:
            singleton_pipeline.append(FeatureExtractor.diameter)

        gather_data = [list(func(data).items())[0] for func in [*singleton_pipeline, *histogram_pipeline]]

        final_dict.update(gather_data)

        return final_dict

    def run_full_pipeline(self, max_num_items=None):

        target_file = self.feature_stats_file
        with jsonlines.open(target_file, mode="a" if self.append_mode else "w") as writer:
            num_full_data = len(self.reader.full_data)
            relevant_subset_of_data = self.reader.full_data[
                                      :min(max_num_items, num_full_data)] if max_num_items else self.reader.full_data
            num_data_being_processed = len(relevant_subset_of_data)
            for item in relevant_subset_of_data:
                del item["poly_data"]
            feature_data_generator = compute_feature_extraction(self, relevant_subset_of_data)
            prepared_data = (FeatureExtractor.jsonify(item) for item in feature_data_generator)
            prepared_data = (dict(timestamp=self.timestamp, **item) for item in prepared_data)
            for next_feature_set in tqdm(prepared_data, total=num_data_being_processed):
                writer.write(next_feature_set)

    @staticmethod
    def jsonify(item):
        result = {key: list(value) if type(value) in [np.ndarray] else value for key, value in item.items()}
        return result

    @staticmethod
    @exception_catcher
    def compactness(data):
        mesh = data["poly_data"]
        edges = mesh.extract_feature_edges(feature_edges=False, manifold_edges=False)
        if edges.n_faces > 0:
            mesh = mesh.fill_holes(1000)  # TODO: Maybe try pip install pymeshfix
        volume = mesh.volume
        cell_ids = PSBDataset._get_cells(mesh)
        cell_areas = PSBDataset._get_cell_areas(mesh.points, cell_ids)
        surface_area = sum(cell_areas)
        compactness = np.power(surface_area, 3) / np.square(volume)
        return {"compactness": compactness}

    @staticmethod
    @exception_catcher
    def sphericity(data):
        mesh = data["poly_data"]
        edges = mesh.extract_feature_edges(feature_edges=False, manifold_edges=False)
        if edges.n_faces > 0:
            mesh = mesh.fill_holes(1000)  # TODO: Maybe try pip install pymeshfix
        volume = mesh.volume
        cell_ids = PSBDataset._get_cells(mesh)
        cell_areas = PSBDataset._get_cell_areas(mesh.points, cell_ids)
        surface_area = sum(cell_areas)
        sphericity_result = (np.power(np.pi, 1 / 3) * np.power(6 * volume, 2 / 3)) / surface_area
        return {"sphericity": sphericity_result}

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
        return {"eccentricity": np.max(eigenvalues) / np.min(eigenvalues)}

    @staticmethod
    @exception_catcher
    def cube_root_volume_four_rand_verts(data):
        # https://stackoverflow.com/a/9866530
        def treaeder_volume(points):
            a, b, c, d = points
            return np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6

        mesh = data["poly_data"]
        random_indices = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1,
                                                               (FeatureExtractor.number_vertices_sampled, 4))
        quad_points = mesh.points[random_indices, :]
        # volumes = np.array([treaeder_volume(points) for points in quad_points])
        A = quad_points[:, 0] - quad_points[:, 3]
        B = np.cross(quad_points[:, 1] - quad_points[:, 3], quad_points[:, 2] - quad_points[:, 3])
        f_volumes = np.abs(np.einsum("ij,ij -> i", A, B)) / 6
        cube_root = f_volumes ** (1 / 3)
        histogram = FeatureExtractor.make_bins(cube_root, FeatureExtractor.number_bins)

        return {"cube_root_volume_four_rand_verts": histogram}

    @staticmethod
    @exception_catcher
    def angle_three_rand_verts(data):
        # This question quite fitted the case (https://bit.ly/3in7MjH)
        angles_degrees = []
        mesh = data["poly_data"]
        indices_triplets = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1,
                                                                 (FeatureExtractor.number_vertices_sampled, 3))
        verts_triplets = [mesh.points[triplet] for triplet in indices_triplets]

        for verts_triplet in verts_triplets:
            p_1, p_2, p_3 = verts_triplet
            p2_1 = p_1 - p_2
            p2_3 = p_3 - p_2
            cosine_angle = np.dot(p2_1, p2_3) / (np.linalg.norm(p2_1) * np.linalg.norm(p2_3))
            angle_radians = np.arccos(cosine_angle)
            angles_degrees.append(np.degrees(angle_radians))

        return {"rand_angle_three_verts": FeatureExtractor.make_bins(angles_degrees, FeatureExtractor.number_bins)}

    @staticmethod
    @exception_catcher
    def dist_two_rand_verts(data):
        distances = []
        mesh = data["poly_data"]
        indices_tuples = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1,
                                                               (FeatureExtractor.number_vertices_sampled, 2))
        verts_tuples = [mesh.points[tup] for tup in indices_tuples]

        for verts_tuple in verts_tuples:
            distance = np.abs(np.diff(verts_tuple, axis=0))
            distances.append(np.linalg.norm(distance))

        return {"rand_dist_two_verts": FeatureExtractor.make_bins(distances, FeatureExtractor.number_bins)}

    @staticmethod
    @exception_catcher
    def dist_bar_vert(data):
        distances = []
        mesh = data["poly_data"]
        bary_center = data["bary_center"]
        indices = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1,
                                                        (FeatureExtractor.number_vertices_sampled, 1))
        rand_verts = mesh.points[indices]
        for vert in rand_verts:
            distance = np.abs(np.diff(np.vstack((bary_center, vert)), axis=0))
            distances.append(np.linalg.norm(distance))
        return {"dist_bar_vert": FeatureExtractor.make_bins(distances, FeatureExtractor.number_bins)}

    @staticmethod
    @exception_catcher
    def dist_sqrt_area_rand_triangle(data):
        mesh = data["poly_data"]
        verts_list = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1, [100, 3])
        triangle_areas = PSBDataset._get_cell_areas(mesh.points, verts_list)
        sqrt_areas = np.sqrt(triangle_areas)
        return {"sqrt_area_rand_three_verts": FeatureExtractor.make_bins(sqrt_areas, FeatureExtractor.number_bins)}

    @staticmethod
    def make_bins(data, n_bins):
        bins = np.linspace(np.min(data), np.max(data), n_bins)
        indices = np.digitize(data, bins)
        count_dict = dict(sorted(Counter(indices).items()))
        count_dict_without_holes = {idx: count_dict[idx] if idx in count_dict.keys() else 0 for idx in
                                    range(1, FeatureExtractor.number_bins + 1)}
        result = np.array(list(count_dict_without_holes.values()))
        return result / result.sum()

    @staticmethod
    def generate_random_ints(min_val, max_val, shape):
        return np.array([np.random.choice(line, shape[1], replace=False) for line in
                         np.repeat(np.arange(min_val, max_val), shape[0], axis=0).reshape(max_val, -1).T])


if __name__ == "__main__":
    FE = FeatureExtractor(DATA_PATH_NORMED_SUBSET)
    pprint(FE.mono_run_pipeline(FE.full_data[0]))
