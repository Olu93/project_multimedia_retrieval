from collections import Counter
from itertools import product

import numpy as np
from tqdm import tqdm
from helper.diameter_computer import compute_diameter
from helper.config import DATA_PATH_NORMED, DEBUG, DATA_PATH_NORMED_SUBSET
from reader import PSBDataset

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

    def __init__(self):
        self.reader = PSBDataset(search_path=DATA_PATH_NORMED_SUBSET if DEBUG else DATA_PATH_NORMED)
        self.reader.read()
        self.reader.load_files_in_memory()
        self.reader.convert_all_to_polydata()
        self.reader.compute_shape_statistics()
        self.full_data = self.reader.full_data

    @staticmethod
    def mono_run_pipeline(self, data, save=False):
        final_dict = {}
        
        print(result)

    def run_full_pipeline(self, max_num_items=None):
        num_full_data = len(self.reader.full_data)
        relevant_subset_of_data = self.reader.full_data[:min(max_num_items, num_full_data)] if max_num_items else self.reader.full_data
        num_data_being_processed = len(relevant_subset_of_data)
        items_generator = tqdm(relevant_subset_of_data, total=num_data_being_processed)
        self.reader.full_data = list((self.mono_run_pipeline(item) for item in items_generator))

    @staticmethod
    def compactness(data):
        mesh = data["poly_data"]
        edges = mesh.extract_feature_edges(feature_edges=False, manifold_edges=False)
        if edges.n_faces > 0:
                mesh.fill_holes(1000, inplace=True)
        volume = mesh.volume
        cell_ids = PSBDataset._get_cells(mesh)
        cell_areas = PSBDataset._get_cell_areas(mesh.points, cell_ids)
        surface_area = sum(cell_areas)
        compactness = np.power(surface_area, 3) / np.square(volume)
        return {"compactness": compactness}

    @staticmethod
    def sphericity(data):
        mesh = data["poly_data"]
        edges = mesh.extract_feature_edges(feature_edges=False, manifold_edges=False)
        if edges.n_faces > 0:
            mesh.fill_holes(1000, inplace=True)
        volume = mesh.volume
        cell_ids = PSBDataset._get_cells(mesh)
        cell_areas = PSBDataset._get_cell_areas(mesh.points, cell_ids)
        surface_area = sum(cell_areas)
        sphericity = (np.power(np.pi, 1/3) * np.power(6*volume, 2/3)) / surface_area
        return {"sphericity": sphericity}

    @staticmethod
    def diameter(data):
        mesh = data["poly_data"]
        return {"diameter": compute_diameter(mesh)}


    @staticmethod
    def aabb_volume(data):
        mesh = data["poly_data"]
        length_x, length_y, length_z = np.abs(np.diff(np.reshape(mesh.bounds, (3, 2))))
        return {"aabb_volume": (length_x * length_y * length_z)}

    @staticmethod
    def surface_area(data):
        mesh = data["poly_data"]
        cell_ids = PSBDataset._get_cells(mesh)
        cell_areas = PSBDataset._get_cell_areas(mesh.points, cell_ids)
        return {"surface_area": sum(cell_areas)}

    @staticmethod
    def eccentricity(data):
        mesh = data["poly_data"]
        A_cov = np.cov(mesh.points.T)
        eigenvalues, _ = np.linalg.eig(A_cov)
        return {"eccentricity": np.max(eigenvalues) / np.min(eigenvalues)}

    @staticmethod
    def cube_root_volume_four_rand_verts(data):
        # https://stackoverflow.com/a/9866530
        def treaeder_volume(points):
            a, b, c, d = points
            return np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6

        mesh = data["poly_data"]
        random_indices = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1, (self.number_vertices_sampled, 4))
        quad_points = mesh.points[random_indices, :]
        # volumes = np.array([treaeder_volume(points) for points in quad_points])
        A = quad_points[:, 0] - quad_points[:, 3]
        B = np.cross(quad_points[:, 1] - quad_points[:, 3], quad_points[:, 2] - quad_points[:, 3])
        f_volumes = np.abs(np.einsum("ij,ij -> i", A, B)) / 6
        cube_root = f_volumes**(1 / 3)
        histogram = FeatureExtractor.make_bins(cube_root, FeatureExtractor.number_bins)

        return {"cube_root_volume_four_rand_verts": histogram}

    @staticmethod
    def angle_three_rand_verts(data):
        # This question quite fitted the case (https://bit.ly/3in7MjH)
        angles_degrees = []
        mesh = data["poly_data"]
        indices_triplets = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1, (FeatureExtractor.number_vertices_sampled, 3))
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
    def dist_two_rand_verts(data):
        distances = []
        mesh = data["poly_data"]
        indices_tuples = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1, (FeatureExtractor.number_vertices_sampled, 2))
        verts_tuples = [mesh.points[tup] for tup in indices_tuples]

        for verts_tuple in verts_tuples:
            distance = np.abs(np.diff(verts_tuple, axis=0))
            distances.append(np.linalg.norm(distance))

        return {"rand_dist_two_verts": FeatureExtractor.make_bins(distances, FeatureExtractor.number_bins)}

    @staticmethod
    def dist_bar_vert(data):
        distances = []
        mesh = data["poly_data"]
        bary_center = data["bary_center"]
        indices = FeatureExtractor.generate_random_ints(0, len(mesh.points) - 1, (FeatureExtractor.number_vertices_sampled, 1))
        rand_verts = mesh.points[indices]
        for vert in rand_verts:
            distance = np.abs(np.diff(np.vstack((bary_center, vert)), axis=0))
            distances.append(np.linalg.norm(distance))
        return {"dist_bar_vert": FeatureExtractor.make_bins(distances, FeatureExtractor.number_bins)}

    @staticmethod
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
        count_dict_without_holes = {idx: count_dict[idx] if idx in count_dict.keys() else 0 for idx in range(1, FeatureExtractor.number_bins + 1)}
        result = np.array(list(count_dict_without_holes.values()))
        return result / result.sum()

    @staticmethod
    def generate_random_ints(min_val, max_val, shape):
        return np.array([np.random.choice(line, shape[1], replace=False) for line in np.repeat(np.arange(min_val, max_val), shape[0], axis=0).reshape(max_val, -1).T])


    


if __name__ == "__main__":
    FE = FeatureExtractor()
    FE.run_full_pipeline(10 if DEBUG else None, save=True)
