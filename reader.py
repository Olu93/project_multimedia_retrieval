import glob
import inspect
import io
import json
from collections import Counter
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from plyfile import PlyData
from tqdm import tqdm

# from helper.config import STAT_PATH, CLASS_FILE, DATA_PATH_PSB
from helper.mp_functions import compute_read
from helper.skeleton import extract_graphical_forms


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

    def run_full_pipeline(self, max_num_items=None):
        self.read()
        num_full_data = len(self.data_descriptors)
        relevant_subset_of_data = self.data_descriptors[:min(max_num_items, num_full_data)] if max_num_items else self.data_descriptors
        num_data_being_processed = len(relevant_subset_of_data)
        read_data = compute_read(self, tqdm(relevant_subset_of_data, total=num_data_being_processed))
        self.full_data = list(read_data)
        return self.full_data

    def run_full_pipeline_slow(self, max_num_items=None):
        self.read()
        num_full_data = len(self.data_descriptors)
        relevant_subset_of_data = self.data_descriptors[:min(max_num_items, num_full_data)] if max_num_items else self.data_descriptors
        num_data_being_processed = len(relevant_subset_of_data)
        read_data = (self.mono_run_pipeline(item) for item in tqdm(relevant_subset_of_data, total=num_data_being_processed))
        self.full_data = list(read_data)
        return self.full_data

    @staticmethod
    def mono_run_pipeline(descriptor):
        data = DataSet.descriptor_to_meta(descriptor)
        data = DataSet.load_mesh(data)
        data = DataSet.triangulate(data)
        data = DataSet.get_base_characteristics(data)
        data = DataSet.get_cell_characteristics(data)
        return data

    @staticmethod
    def descriptor_to_meta(data):
        return {"meta_data": data}

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
        if np.sum(cell_areas) == 0:
            print(f'{item["meta_data"]["path"]} has no surface area!')
        item["bary_center"] = np.array(DataSet._compute_center(cell_centers, cell_areas))
        item["statistics"].update(dict(zip(["bound_" + b for b in "xmin xmax ymin ymax zmin zmax".split()], mesh.bounds)))
        item["statistics"].update({f"center_{dim}": val for dim, val in zip("x y z".split(), item["bary_center"])})
        item["statistics"]["cell_area_mean"] = np.mean(cell_areas)
        item["statistics"]["cell_area_std"] = np.std(cell_areas)
        return item

    @staticmethod
    def load_mesh(descriptor):
        """
        Loads meshes from different file types and computes basic operations
        """
        mesh = None
        file_name = descriptor["meta_data"]["path"]
        if not file_name:
            return mesh
        if str(file_name).split(".")[1] != "off":
            mesh = DataSet._load_ply(file_name)
        elif str(file_name).split(".")[1] == "off":
            mesh = DataSet._load_off(file_name)
        else:
            raise Exception("File type not yet supported.")
        descriptor["data"] = mesh
        return descriptor

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

    def load_image_data(self):
        print("Load images")
        len_of_ds = len(self.data_descriptors)
        self.full_data = [
            dict(**mesh_data, images=extract_graphical_forms(pv.PolyData(mesh_data["data"]["vertices"], mesh_data["data"]["faces"])))
            for mesh_data in tqdm(self.full_data, total=len_of_ds)
        ]

    def compute_shape_statistics(self):
        self.all_statistics = pd.DataFrame([mesh_object["statistics"] for mesh_object in self.full_data])
        self.has_stats = True
        print(f"Finished {inspect.currentframe().f_code.co_name}")

    def save_statistics(self, stats_path=None, stats_filename=None):
        path_for_statistics = stats_path if stats_path else self.stats_path
        assert path_for_statistics, "No path for statistics given. Either set it for specific class or provide it as param!"
        self.all_statistics.to_csv(str(path_for_statistics) + f"/{stats_filename if stats_filename else 'statistics.csv'}", index=False)
        print(f"Finished {inspect.currentframe().f_code.co_name}")

    def convert_all_to_polydata(self):
        self.full_data = [dict(**mesh_data, poly_data=pv.PolyData(mesh_data["data"]["vertices"], mesh_data["data"]["faces"])) for mesh_data in self.full_data]
        self.has_poly_data = True
        print(f"Finished {inspect.currentframe().f_code.co_name}")

    def detect_outliers(self):
        assert self.has_stats, f"No statistics were computed. Run {self.compute_shape_statistics.__name__} first"

        def add_to_stats(m_object):
            m_object["statistics"]["faces_outlier"] = m_object["faces_outlier"]
            m_object["statistics"]["vertices_outlier"] = m_object["vertices_outlier"]
            m_object["statistics"]["face_area_outlier"] = m_object["face_area_outlier"]
            return m_object

        all_faces_counts, all_vertices_counts = self.all_statistics["faces"], self.all_statistics["vertices"]

        self.faces_mean, self.faces_std = all_faces_counts.mean(), all_faces_counts.std()
        self.vertices_mean, self.vertices_std = all_vertices_counts.mean(), all_vertices_counts.std()
        self.face_area_mean, self.face_area_std = self.all_statistics["cell_area_mean"].mean(), self.all_statistics["cell_area_std"].mean()

        init_pipe = (mesh_object for mesh_object in self.full_data)
        faces_pipe = (dict(**mesh_object, faces_outlier=True if np.abs(mesh_object["statistics"]["faces"] - self.faces_mean) > 2 * self.faces_std else False)
                      for mesh_object in init_pipe)
        vertices_pipe = (dict(**mesh_object, vertices_outlier=True if np.abs(mesh_object["statistics"]["vertices"] - self.vertices_mean) > 2 * self.vertices_std else False)
                         for mesh_object in faces_pipe)
        face_area_pipe = (dict(**mesh_object,
                               face_area_outlier=True if np.abs(mesh_object["statistics"]["cell_area_mean"] - self.face_area_mean) > 2 * self.face_area_std else False)
                          for mesh_object in vertices_pipe)

        add_to_stats_pipe = (add_to_stats(mesh_object) for mesh_object in face_area_pipe)

        self.full_data = list(add_to_stats_pipe)
        self.all_statistics = pd.DataFrame([mesh_object["statistics"] for mesh_object in self.full_data])
        print(f"Finished {inspect.currentframe().f_code.co_name}")

    @staticmethod
    def _read(file_path):
        path = Path(file_path)
        file_name = path.stem
        file_type = path.suffix
        label = "no_class"
        coarse_label = "no_coarse_class"
        meta_data = {"label": label, "label_coarse": coarse_label, "name": file_name, "type": file_type, "path": path.resolve().as_posix()}
        data = DataSet._load_mesh(meta_data["path"])
        poly_data = pv.PolyData(data["vertices"], data["faces"])
        curr_data = dict(meta_data=meta_data, data=data, poly_data=poly_data)
        statistics = DataSet._compute_statistics(curr_data)
        final_data = dict(curr_data, statistics=statistics)
        return final_data

    @staticmethod
    def _compute_statistics(mesh):
        poly_data_object = mesh["poly_data"]
        triangulized_poly_data_object = poly_data_object.triangulate()
        mesh["poly_data"] = triangulized_poly_data_object
        statistics = {"id": mesh["meta_data"]["name"], "label": mesh["meta_data"]["label"], "faces": poly_data_object.n_faces, "vertices": poly_data_object.n_points}
        statistics.update(dict(zip(["bound_" + b for b in "xmin xmax ymin ymax zmin zmax".split()], poly_data_object.bounds)))
        cell_ids = DataSet._get_cells(mesh["poly_data"])
        cell_point_counts = [len(cell) for cell in cell_ids]
        cell_counter = Counter(cell_point_counts)
        statistics.update({f"cell_type_{k}": v for k, v in cell_counter.items()})

        cell_areas = DataSet._get_cell_areas(triangulized_poly_data_object.points, cell_ids)
        cell_centers = triangulized_poly_data_object.cell_centers().points
        mesh["bary_center"] = np.array(DataSet._compute_center(cell_centers, cell_areas))
        statistics.update({f"center_{dim}": val for dim, val in zip("x y z".split(), mesh["bary_center"])})

        statistics["cell_area_mean"] = np.mean(cell_areas)
        statistics["cell_area_std"] = np.std(cell_areas)
        return statistics

    @staticmethod
    def _get_cell_areas(mesh_vertices, mesh_cells):
        cell_combinations = mesh_vertices[mesh_cells, :]
        cell_areas_fast = np.abs(np.linalg.norm(np.cross((cell_combinations[:, 0] - cell_combinations[:, 1]),
                                                         (cell_combinations[:, 0] - cell_combinations[:, 2])), axis=1)) / 2  # https://math.stackexchange.com/a/128999

        return cell_areas_fast

    @staticmethod
    def _get_cells(mesh):
        """Returns a list of the cells from this mesh.
        This properly unpacks the VTK cells array.
        There are many ways to do this, but this is
        safe when dealing with mixed cell types."""

        return mesh.faces.reshape(-1, 4)[:, 1:4]

    @staticmethod
    def _load_mesh(file_name):
        """
        Loads meshes from different file types and computes basic operations
        """
        mesh = None
        if not file_name:
            return mesh
        if str(file_name)[::-1].split(".")[0][::-1] != "off":
            mesh = DataSet._load_ply(file_name)
        elif str(file_name)[::-1].split(".")[0][::-1] == "off":
            mesh = DataSet._load_off(file_name)
        else:
            raise Exception("File type not yet supported.")
        return mesh

    @staticmethod
    def _load_ply(file):
        try:
            ply_data = PlyData.read(file)
            vertices = np.array(ply_data["vertex"].data.tolist())
            faces = ply_data["face"]["vertex_indices"].tolist()
            faces_with_number_of_faces = [np.hstack([face.shape[0], face]) for face in faces]
            flattened_faces = np.hstack(faces_with_number_of_faces).flatten()
            return {"vertices": vertices, "faces": flattened_faces}
        except Exception as e:
            print(f"ERROR: Couldn't load {file}")
            return None

    @staticmethod
    def _load_off(file):
        try:
            off_data = pv.read(file)
            faces = off_data.cells
            vertices = np.array(off_data.points)
            return {"vertices": vertices, "faces": faces}
        except Exception as e:
            print(f"ERROR {e}: Couldn't load {file}")
            return None

    @staticmethod
    def _extract_descr(file_path):
        path = Path(file_path)
        file_name = path.stem
        file_type = path.suffix
        label = "no_class"
        return {"label": label, "name": file_name, "type": file_type, "path": path.resolve().as_posix()}

    @staticmethod
    def _compute_center(face_normals, face_areas):
        weighted_normals = face_areas.reshape(-1, 1) * face_normals
        bary_center = np.sum(weighted_normals, axis=0) / np.sum(face_areas)
        return bary_center

    def show_class_histogram(self):
        pd_data = pd.DataFrame(self.data_descriptors)
        counts_list = list(Counter(pd_data["label"].astype(str).values).items())
        counts = pd.DataFrame([{"label": label, "counts": int(count)} for label, count in counts_list])
        plt.bar(counts["label"], counts["counts"])
        plt.xticks(rotation=90)
        plt.show()

    @staticmethod
    def show_barycenter(mesh_item):
        plotter = pv.Plotter()
        some_sphere = pv.Sphere(radius=0.001)
        some_sphere.translate(mesh_item["bary_center"])
        plotter.add_mesh(some_sphere)
        plotter.add_mesh(mesh_item["poly_data"])
        plotter.show()


class PSBDataset(DataSet):
    def __init__(self, search_path=None, stats_path=None, **kwargs):
        with open('config.json') as f:
            data = json.load(f)
        self.search_path = data["DATA_PATH_PSB"] if not search_path else search_path
        self.search_paths = [Path(self.search_path) / scheme for scheme in self.schemes]
        self.stats_path = data["STAT_PATH"] if not stats_path else Path(stats_path)
        self.class_member_ships = PSBDataset.load_classes(kwargs.get("class_file_path", data["CLASS_FILE"]))
        self.class_member_ships_coarse = PSBDataset.load_classes(kwargs.get("class_file_path_coarse", data["CLASS_FILE_COARSE"]))
        super().__init__(self.search_paths, self.stats_path)

    @staticmethod
    def load_classes(class_file_path):
        if not class_file_path:
            return {}
        path_to_classes = Path(class_file_path)
        search_pattern = path_to_classes / "*.cla"
        class_files = list(glob.glob(str(search_pattern), recursive=True))
        class_file_handlers = [io.open(cfile, mode="r") for cfile in class_files]
        class_file_content = [cf.read().split("\n") for cf in class_file_handlers]
        curr_class = None
        class_memberships = {}
        for cfile in class_file_content:
            for line in cfile:
                line_content = line.split()
                line_nr_of_items = len(line_content)
                if line_nr_of_items == 2:
                    continue
                if line_nr_of_items == 3:
                    curr_class = line_content[0]
                if line_nr_of_items == 1:
                    class_memberships[f"m{line_content[0]}"] = curr_class
        return class_memberships

    def read(self):
        self.data_descriptors = [self._extract_descr(file_path) for file_path in self.data_file_paths]
        self.has_descriptors = True

    def _extract_descr(self, file_path):
        path = Path(file_path)
        file_name = path.stem
        file_type = path.suffix
        label = self.class_member_ships.get(file_name, "no_class")
        label_coarse = self.class_member_ships_coarse.get(file_name, "no_class")
        return {"label": label, "label_coarse": label_coarse, "name": file_name, "type": file_type, "path": path.resolve().as_posix()}


class ModelNet40Dataset(DataSet):
    def __init__(self, search_path=None, stats_path=None, class_file_path=None):
        self.search_path = "data/mn40" if not search_path else search_path
        self.search_paths = [Path(self.search_path) / scheme for scheme in self.schemes]
        self.stats_path = Path("data/mn40") if not stats_path else Path(stats_path)
        self.class_file_path = class_file_path
        super().__init__(self.search_paths, self.stats_path)

    def read(self):
        self.data_descriptors = [self._extract_descr(file_path) for file_path in self.data_file_paths]
        self.has_descriptors = True

    def _extract_descr(self, file_path):
        path = Path(file_path)
        label = path.parents[1].as_posix().split("/")[-1]
        file_name = path.stem
        file_type = path.suffix
        return {"label": label, "name": file_name, "type": file_type, "path": path.resolve().as_posix()}


if __name__ == "__main__":
    pass
    #dataset = PSBDataset(stats_path=STAT_PATH, search_path=DATA_PATH_PSB, class_file_path=CLASS_FILE)
    #dataset.read()
    #dataset.show_class_histogram()
    #dataset.load_files_in_memory()
    #dataset.convert_all_to_polydata()
    #dataset.compute_shape_statistics()
    #dataset.detect_outliers()
    #dataset.save_statistics()
    # pprint(dataset.full_data[0])