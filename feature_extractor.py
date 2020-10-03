from collections import Counter
from itertools import product

import numpy as np
from tqdm import tqdm

from helper.config import DATA_PATH_NORMED, DEBUG, DATA_PATH_NORMED_SUBSET
from reader import PSBDataset

# TODO: [x] surface area
# TODO: [x] compactness (with respect to a sphere)
# TODO: [x] axis-aligned bounding-box volume
# TODO: [] diameter
# TODO: [x] eccentricity
# TODO: [] A3: angle between 3 random vertices
# TODO: [] D1: distance between barycenter and random vertex
# TODO: [] D2: distance between 2 random vertices
# TODO: [] D3: square root of area of triangle given by 3 random vertices
# TODO: [] D4: cube root of volume of tetrahedron formed by 4 random vertices


class FeatureExtractor:
    def __init__(self):
        self.reader = PSBDataset(search_path=DATA_PATH_NORMED_SUBSET if DEBUG else DATA_PATH_NORMED)
        self.reader.read()
        self.reader.load_files_in_memory()
        self.reader.convert_all_to_polydata()
        self.reader.compute_shape_statistics()
        self.full_data = self.reader.full_data

    def mono_run_pipeline(self, data):
        result = self.diameter(data)
        print(result)

    def run_full_pipeline(self, max_num_items=None):
        num_full_data = len(self.reader.full_data)
        relevant_subset_of_data = self.reader.full_data[:min(max_num_items, num_full_data)] if max_num_items else self.reader.full_data
        num_data_being_processed = len(relevant_subset_of_data)
        items_generator = tqdm(relevant_subset_of_data, total=num_data_being_processed)
        self.reader.full_data = list((self.mono_run_pipeline(item) for item in items_generator))

    def compactness(self, data):
        mesh = data["poly_data"]
        volume = mesh.volume
        cell_ids = self.reader._get_cells(mesh)
        cell_areas = self.reader._get_cell_areas(mesh.points, cell_ids, "")
        surface_area = sum(cell_areas)
        pi = np.pi
        compactness = np.power(surface_area, 3) / (36 * pi * np.square(volume))
        return {"compactness": compactness}

    def diameter(self, data):

        mesh = data["poly_data"]
        all_vertices = np.array(mesh.points)
        vertices1, vertices2 = list(zip(*product(all_vertices, all_vertices)))
        difference_between_points = np.array(vertices1) - np.array(vertices2)
        squared_difference = np.square(difference_between_points)
        sum_of_squared = np.sum(squared_difference, axis=1)
        L2_distance = np.sqrt(sum_of_squared)
        max_distance = np.max(L2_distance)
        return {"diameter": max_distance}

    def aabb_volume(self, data):
        mesh = data["poly_data"]
        length_x, length_y, length_z = np.abs(np.diff(np.reshape(mesh.bounds, (3, 2))))
        return {"aabb_volume": (length_x * length_y * length_z)}

    def surface_area(self, data):
        mesh = data["poly_data"]
        cell_ids = self.reader._get_cells(mesh)
        cell_areas = self.reader._get_cell_areas(mesh.points, cell_ids)
        return {"surface_area": sum(cell_areas)}

    def eccentricity(self, data):
        mesh = data["poly_data"]
        A_cov = np.cov(mesh.points.T)
        eigenvalues, _ = np.linalg.eig(A_cov)
        return {"eccentricity": np.max(eigenvalues) / np.min(eigenvalues)}

    def cube_root_volume_four_rand_verts(self, data):
        # https://stackoverflow.com/a/9866530
        def treaeder_volume(points):
            a, b, c, d = points
            return np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6

        mesh = data["poly_data"]
        random_indices = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1, (100, 4))
        quad_points = mesh.points[random_indices, :]
        # volumes = np.array([treaeder_volume(points) for points in quad_points])
        A = quad_points[:, 0] - quad_points[:, 3]
        B = np.cross(quad_points[:, 1] - quad_points[:, 3], quad_points[:, 2] - quad_points[:, 3])
        f_volumes = np.abs(np.einsum("ij,ij -> i", A, B)) / 6
        cube_root = f_volumes**(1 / 3)
        histogram = FeatureExtractor.make_bins(cube_root, 10)

        return {"cube_root_volume_four_rand_verts": histogram}

    def angle_three_rand_verts(self, dataset):
        # This question quite fitted the case (https://bit.ly/3in7MjH)
        data_out = dict()
        mesh = data["poly_data"]
        name = data["meta_data"]["name"]
        indices_triplets = np.random.randint(0, len(mesh.points) - 1, (100, 3))
        vertices_triplets = [mesh.points[triplet] for idx, triplet in indices_triplets]
        for _ in range(1, 100):
            random_indices = np.random.randint(0, len(mesh.points) - 1, (3, ))  # Sampling
            name = mesh["meta_data"]["name"]
            p_1, p_2, p_3 = mesh["poly_data"].points[random_indices]
            p2_1 = p_1 - p_2
            p2_3 = p_3 - p_2
            cosine_angle = np.dot(p2_1, p2_3) / (np.linalg.norm(p2_1) * np.linalg.norm(p2_3))
            angle_radians = np.arccos(cosine_angle)
            angle_degrees = np.degrees(angle_radians)
            data_out.update({name: {"rand_angle_three_verts": angle_degrees}})
        return data_out

    def dist_two_rand_verts(self, dataset):
        # NOT TESTED
        data_out = dict()
        for mesh in dataset:
            random_indices = np.random.randint(0, len(dataset) - 1, (2, ))
            name = mesh["meta_data"]["name"]
            two_rand_points = mesh["poly_data"].points[random_indices]
            distance = np.abs(np.diff(two_rand_points, axis=0))
            data_out.update({name: {"rand_dist_two_verts": distance}})
        return data_out

    def dist_bar_vert(self, dataset):
        # NOT TESTED
        data_out = dict()
        for mesh in dataset:
            name = mesh["meta_data"]["name"]
            bary_center = mesh["poly_data"]["bary_center"]
            rnd_idx = np.random.randint(0, len(mesh["poly_data"].points))
            rnd_vert = mesh["poly_data"].points[rnd_idx]
            distance = np.abs(np.diff(np.reshape(np.concatenate((bary_center, rnd_vert)), (2, 3)), axis=0))
            data_out.update({name: {"dist_bar_vert": distance}})
        return data_out

    @staticmethod
    def make_bins(data, n_bins):
        bins = np.linspace(np.min(data), np.max(data), n_bins)
        indices = np.digitize(data, bins)
        count_dict = Counter(indices)
        return dict(sorted(count_dict.items()))

    @staticmethod
    def generate_random_ints(min_val, max_val, shape):
        return np.array([np.random.choice(line, shape[1], replace=False) for line in np.repeat(np.arange(min_val, max_val), shape[0], axis=0).reshape(max_val, -1).T])


if __name__ == "__main__":
    FE = FeatureExtractor()
    FE.run_full_pipeline(10 if DEBUG else None)
