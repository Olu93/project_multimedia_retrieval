import glob
import inspect
import io
from collections import Counter
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from plyfile import PlyData
from tqdm import tqdm

from helper.config import STAT_PATH, CLASS_FILE, DATA_PATH_DEBUG, DATA_PATH_PSB


class DataSet:
    data_descriptors = []
    full_data = []
    stats_path = None
    schemes = ["**/*.off", "**/*.ply"]
    has_descriptors = None
    has_stats = False
    has_loaded_data = False
    has_poly_data = False
    has_outliers = False
    class_member_ships = {}

    def __init__(self, search_paths=[], stats_path=None):
        data_folder = [Path(path) for path in search_paths]
        self.all_statistics = None
        self.stats_path = stats_path
        self.data_file_paths = list(chain(*[glob.glob(str(path), recursive=True) for path in data_folder]))

    @staticmethod
    def mono_run_pipeline(descriptor):
        data = DataSet._load_mesh(descriptor)
        data = DataSet.triangulate(data)
        data = DataSet.get_base_characteristics(data)
        data = DataSet.get_cell_characteristics(data)
        return data

    @staticmethod
    def get_base_characteristics(item):
        poly_data_object = pv.PolyData(item["data"]["vertices"], item["data"]["faces"])
        item["statistics"] = {"id": item["meta_data"]["name"], "label": item["meta_data"]["label"], "faces": poly_data_object.n_faces, "vertices": poly_data_object.n_points}
        return item

    @staticmethod
    def triangulate(item):
        mesh = pv.PolyData(item["data"]["vertices"], item["data"]["faces"])
        item["data"]["vertices"] = mesh.points
        item["data"]["faces"] = mesh.faces
        return item

    @staticmethod
    def get_cell_characteristics(item):
        mesh = pv.PolyData(item["data"]["vertices"], item["data"]["faces"])
        cell_ids = DataSet._get_cells(mesh)
        cell_areas = DataSet._get_cell_areas(mesh.points, cell_ids)
        cell_centers = mesh.cell_centers().points
        item["bary_center"] = np.array(DataSet._compute_center(cell_centers, cell_areas))
        item["statistics"].update(dict(zip(["bound_" + b for b in "xmin xmax ymin ymax zmin zmax".split()], mesh.bounds)))
        item["statistics"].update({f"center_{dim}": val for dim, val in zip("x y z".split(), item["bary_center"])})
        item["statistics"]["cell_area_mean"] = np.mean(cell_areas)
        item["statistics"]["cell_area_std"] = np.std(cell_areas)
        return item

    def run_full_pipeline(self, limit):
        pass

    @staticmethod
    def _read(file_path):
        path = Path(file_path)
        file_name = path.stem
        file_type = path.suffix
        label = "no_class"
        meta_data = {"label": label, "name": file_name, "type": file_type, "path": path.resolve().as_posix()}
        data = DataSet._load_mesh(meta_data["path"])
        poly_data = pv.PolyData(data["vertices"], data["faces"])
        curr_data = dict(meta_data=meta_data, data=data, poly_data=poly_data)
        statistics = DataSet._compute_statistics(curr_data)
        final_data = dict(curr_data, statistics=statistics)
        return final_data

    @staticmethod
    def load_mesh(file_name):
        """
        Loads meshes from different file types and computes basic operations
        """
        mesh = None
        if not file_name:
            return mesh
        if str(file_name).split(".")[1] != "off":
            mesh = DataSet._load_ply(file_name)
        elif str(file_name).split(".")[1] == "off":
            mesh = DataSet._load_off(file_name)
        else:
            raise Exception("File type not yet supported.")
        return mesh

    def load_files_in_memory(self):
        assert self.has_descriptors, f"Dunno the file locations. Run {self.read.__name__} function first."
        len_of_ds = len(self.data_descriptors)
        print(f"Loading {len_of_ds} models into memory!")
        full_data_generator = ({
            "meta_data": file,
            "data": self._load_ply(file["path"]) if not file["type"] == ".off" else self._load_off(file["path"])
        } for file in tqdm(self.data_descriptors, total=len_of_ds))
        self.full_data = [item for item in list(full_data_generator) if item["data"]]
        self.has_loaded_data = True
        print(f"Finished {inspect.currentframe().f_code.co_name}")

    def compute_shape_statistics(self):
        assert self.has_poly_data, f"No pyvista objects available. Run {self.convert_all_to_polydata.__name__} first"
        self.full_data = [dict(object_descriptor, statistics=self._compute_statistics(object_descriptor)) for object_descriptor in self.full_data]
        self.all_statistics = pd.DataFrame([mesh_object["statistics"] for mesh_object in self.full_data])
        self.has_stats = True
        print(f"Finished {inspect.currentframe().f_code.co_name}")