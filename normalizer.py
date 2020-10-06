import os
from os import path
from pathlib import Path

import numpy as np
import pyacvd
from pyvista import PolyData
import pyvista as pv
from tqdm import tqdm

from helper.config import DEBUG, DATA_PATH_PSB, DATA_PATH_DEBUG, CLASS_FILE, DATA_PATH_NORMED
from reader import PSBDataset
from helper.mp_functions import compute_normalization


class Normalizer:
    def __init__(self, reader=None):
        self.num_avg_verts = 35000
        if reader:
            self.reader = reader
            self.reader.read()
            self.reader.load_files_in_memory()
            self.reader.convert_all_to_polydata()
            self.reader.compute_shape_statistics()
            self.full_data = self.reader.full_data
            # self.history = []
            # self.history.append([item["poly_data"] for item in self.full_data])

    # https://www.grasshopper3d.com/forum/topics/best-uniform-remesher-for-patterning-organic-suraces
    def uniform_remeshing(self):
        tmp_mesh = []
        print("Remeshing")
        for mesh, data in zip(self.history[-1], self.full_data):
            print(f"Processing: {data['meta_data']['name']}")
            data = mesh.clean()
            clus = pyacvd.Clustering(data)
            while len(clus.mesh.points) < 30000:
                clus.subdivide(2)
            clus.cluster(10000)
            remesh = clus.create_mesh()
            tmp_mesh.append(remesh)
        self.history.append(tmp_mesh)

    @staticmethod
    def mono_uniform_remeshing(data):
        print(f"Remeshing: {data['meta_data']['name']}")
        data = data["poly_data"].clean()
        clus = pyacvd.Clustering(data)
        try:
            while len(clus.mesh.points) < 30000:
                clus.subdivide(2)
        except MemoryError as e:
            print(f"Ups that a little too much memory! {e}")
        clus.cluster(10000)
        remesh = clus.create_mesh()
        return remesh

    def center(self):
        print("Centering")
        tmp_mesh = []
        for mesh, full_mesh_stuff in zip(self.history[-1], self.full_data):
            remesh = PolyData(mesh.points.copy(), mesh.faces.copy())
            offset = full_mesh_stuff["bary_center"]
            remesh.translate(np.zeros_like(offset) - offset)
            tmp_mesh.append(remesh)
        self.history.append(tmp_mesh)

    @staticmethod
    def mono_centering(data):
        print(f"Centering: {data['meta_data']['name']}")
        mesh = data["poly_data"]
        remesh = PolyData(mesh.points.copy(), mesh.faces.copy())
        offset = data["bary_center"]
        remesh.translate(np.zeros_like(offset) - offset)
        return remesh

    def align(self):
        print("Aligning")
        tmp_mesh = []
        for mesh in self.history[-1]:
            A_cov = np.cov(mesh.points.T)
            eigenvalues, eigenvectors = np.linalg.eig(A_cov)
            biggest_idx = np.argsort(-eigenvalues)
            biggest_vec = eigenvectors[:, biggest_idx]
            new_points = np.dot(mesh.points, biggest_vec)
            remesh = PolyData(new_points, mesh.faces.copy())
            tmp_mesh.append(remesh)
        self.history.append(tmp_mesh)

    @staticmethod
    def mono_alignment(data):
        print(f"Aligning: {data['meta_data']['name']}")
        mesh = data["poly_data"]
        A_cov = np.cov(mesh.points.T)
        eigenvalues, eigenvectors = np.linalg.eig(A_cov)
        biggest_idx = np.argsort(-eigenvalues)
        biggest_vec = eigenvectors[:, biggest_idx]
        new_points = np.dot(mesh.points, biggest_vec)
        remesh = PolyData(new_points, mesh.faces.copy())
        return remesh

    def scale_to_union(self):
        print("\nScaling")
        tmp_mesh = []
        for mesh in self.history[-1]:
            max_range = np.max(mesh.points, axis=0)
            min_range = np.min(mesh.points, axis=0)
            lengths_range = max_range - min_range
            longest_range = np.max(lengths_range)
            scaled_points = (mesh.points - min_range) / longest_range
            remesh = PolyData(scaled_points, mesh.faces.copy())
            tmp_mesh.append(remesh)
        self.history.append(tmp_mesh)

    @staticmethod
    def mono_scaling(data):
        print(f"\nScaling: {data['meta_data']['name']}")
        mesh = data["poly_data"]
        max_range = np.max(mesh.points, axis=0)
        min_range = np.min(mesh.points, axis=0)
        lengths_range = max_range - min_range
        longest_range = np.max(lengths_range)
        scaled_points = (mesh.points - min_range) / longest_range
        remesh = PolyData(scaled_points, mesh.faces.copy())
        return remesh

    def flipping(self):
        tmp_mesh = []
        for mesh in self.history[-1]:
            face_centers = mesh.cell_centers().points
            overall_signs = np.sum(np.sign(face_centers) * np.square(face_centers), axis=0)
            flipped_points = np.sign(overall_signs) * mesh.points
            remesh = PolyData(flipped_points, mesh.faces.copy())
            tmp_mesh.append(remesh)
        self.history.append(tmp_mesh)

    @staticmethod
    def mono_flipping(data):
        print(f"Flipping: {data['meta_data']['name']}")
        mesh = data["poly_data"]
        face_centers = mesh.cell_centers().points
        overall_signs = np.sum(np.sign(face_centers) * np.square(face_centers), axis=0)
        flipped_points = np.sign(overall_signs) * mesh.points
        remesh = PolyData(flipped_points, mesh.faces.copy())
        return remesh

    def save_dataset(self):
        for processed_mesh, data in zip(self.history[-1], self.full_data):
            target_directory = f"processed_data\\{data['meta_data']['label']}"
            print(f"Writing {data['meta_data']['name']}.ply to {target_directory}")
            if not path.exists(target_directory):
                os.mkdir(target_directory)
            processed_mesh.save(f"{target_directory}\\{data['meta_data']['name']}.ply")

    @staticmethod
    def mono_saving(data):
        print(f"Saving: {data['meta_data']['name']}")
        mesh = pv.PolyData(data["data"]["vertices"], data["data"]["faces"])
        target_directory = Path(f"{DATA_PATH_NORMED}/{data['meta_data']['label']}")
        final_directory = target_directory / f"{data['meta_data']['name']}.ply"

        print(f"Writing {data['meta_data']['name']}.ply to {target_directory}")
        if not path.exists(target_directory):
            print(f"Creating path {target_directory} for saving {data['meta_data']['name']}")
            os.makedirs(target_directory)

        mesh.save(final_directory)
        print(f"{data['meta_data']['name']} was {'successfully saved.' if path.exists(final_directory) else 'NOT saved!'}")
        return data

    @staticmethod
    def mono_run_pipeline(data):
        new_mesh = pv.PolyData(data["data"]["vertices"], data["data"]["faces"])
        history = [{"op": "(a) Original", "data": new_mesh}]
        new_mesh = Normalizer.mono_scaling(dict(data, poly_data=new_mesh))
        history.append({"op": "(b) Scale", "data": new_mesh})
        new_mesh = Normalizer.mono_centering(dict(data, poly_data=new_mesh))
        history.append({"op": "(c) Center", "data": new_mesh})
        new_mesh = Normalizer.mono_alignment(dict(data, poly_data=new_mesh))
        history.append({"op": "(d) Align", "data": new_mesh})
        new_mesh = Normalizer.mono_flipping(dict(data, poly_data=new_mesh))
        history.append({"op": "(e) Flip", "data": new_mesh})
        new_mesh = Normalizer.mono_uniform_remeshing(dict(data, poly_data=new_mesh))
        history.append({"op": "(f) Remesh", "data": new_mesh})

        history = [{"op": step["op"], "data": (step["data"].points, step["data"].faces)} for step in history]
        print(f"Pipeline complete for {data['meta_data']['name']}")
        return dict(data, history=history)

    def run_full_pipeline(self, max_num_items=None):
        num_full_data = len(self.reader.full_data)
        relevant_subset_of_data = self.reader.full_data[:min(max_num_items, num_full_data)] if max_num_items else self.reader.full_data
        num_data_being_processed = len(relevant_subset_of_data)
        for item in relevant_subset_of_data:
            del item["poly_data"]
        normalization_data_generator = compute_normalization(self, relevant_subset_of_data)
        items_generator = tqdm(normalization_data_generator, total=num_data_being_processed)
        self.reader.full_data = list((self.mono_saving(item) for item in items_generator))
        print("Done!")


if __name__ == '__main__':
    norm = Normalizer(PSBDataset(DATA_PATH_DEBUG if DEBUG else DATA_PATH_PSB, class_file_path=CLASS_FILE))
    norm.run_full_pipeline(10 if DEBUG else None)
    print("Done")
