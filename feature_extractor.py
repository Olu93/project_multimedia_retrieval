from tqdm import tqdm

from helper.config import DATA_PATH_NORMED
from reader import PSBDataset
import pyvista as pv
import numpy as np


class FeatureExtractor:
    def __init__(self):
        self.reader = PSBDataset(search_path=DATA_PATH_NORMED)
        self.reader.read()
        self.reader.load_files_in_memory()
        self.reader.convert_all_to_polydata()
        self.reader.compute_shape_statistics()
        self.full_data = self.reader.full_data

    def compactness(self, data):
        mesh = data["poly_data"]
        volume = mesh.volume
        cell_ids = self.reader._get_cells(mesh)
        cell_areas = self.reader._get_cell_areas(mesh.points, cell_ids, "")
        surface_area = sum(cell_areas)
        pi = np.pi
        compactness = np.power(surface_area, 3)/(36*pi*np.square(volume))
        return {"compactness": compactness}



    def mono_run_pipeline(self, data):
        pass

    def run_full_pipeline(self, max_num_items=None):
        num_full_data = len(self.reader.full_data)
        relevant_subset_of_data = self.reader.full_data[
                                  :min(max_num_items, num_full_data)] if max_num_items else self.reader.full_data
        num_data_being_processed = len(relevant_subset_of_data)
        items_generator = tqdm(relevant_subset_of_data, total=num_data_being_processed)
        self.reader.full_data = list((self.mono_run_pipeline(item) for item in items_generator))

if __name__ == '__main__':
    FExtractor = FeatureExtractor()
    FExtractor.compactness(pv.Cube().triangulate())
    print("Done")
