import numpy as np
from tqdm import tqdm
from itertools import product
from helper.config import DATA_PATH_NORMED, DEBUG, DATA_PATH_NORMED_SUBSET
from helper.mp_functions import compute_distance
from reader import PSBDataset
import multiprocessing as mp
import math

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


if __name__ == "__main__":
    FE = FeatureExtractor()
    FE.run_full_pipeline(10 if DEBUG else None)
