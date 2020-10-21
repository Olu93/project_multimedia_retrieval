import os
from os import path
from pathlib import Path

import numpy as np
import pyacvd
import pyvista as pv
from pyvista import PolyData
from tqdm import tqdm

from helper.config import DEBUG, DATA_PATH_PSB, DATA_PATH_DEBUG, CLASS_FILE
from helper.mp_functions import compute_normalization
from reader import PSBDataset

VERBOSE = False


class Normalizer:
    def __init__(self, reader=None, target_path="processed_data"):
        self.num_avg_verts = 35000
        self.target_path = target_path
        if reader:
            self.reader = reader
            self.full_data = reader.run_full_pipeline()
            # self.history = []
            # self.history.append([item["poly_data"] for item in self.full_data])

    # https://www.grasshopper3d.com/forum/topics/best-uniform-remesher-for-patterning-organic-suraces
    @staticmethod
    def mono_uniform_remeshing(data):
        if VERBOSE: print(f"Remeshing: {data['meta_data']['name']}")
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
        if VERBOSE: print("Centering")
        tmp_mesh = []
        for mesh, full_mesh_stuff in zip(self.history[-1], self.full_data):
            remesh = PolyData(mesh.points.copy(), mesh.faces.copy())
            offset = mesh.center
            remesh.translate(np.zeros_like(offset) - offset)
            tmp_mesh.append(remesh)
        self.history.append(tmp_mesh)

    @staticmethod
    def mono_centering(data):
        if VERBOSE: print(f"Centering: {data['meta_data']['name']}")
        mesh = data["poly_data"]
        remesh = PolyData(mesh.points.copy(), mesh.faces.copy())
        offset = mesh.center
        remesh.translate(np.zeros_like(offset) - offset)
        return remesh

    @staticmethod
    def mono_alignment(data):
        if VERBOSE: print(f"Aligning: {data['meta_data']['name']}")
        mesh = data["poly_data"]
        A_cov = np.cov(mesh.points.T)
        eigenvalues, eigenvectors = np.linalg.eig(A_cov)
        biggest_idx = np.argsort(-eigenvalues)
        biggest_vec = eigenvectors[:, biggest_idx]
        new_points = np.dot(mesh.points, biggest_vec)
        remesh = PolyData(new_points, mesh.faces.copy())
        return remesh

    @staticmethod
    def mono_scaling(data):
        if VERBOSE: print(f"\nScaling: {data['meta_data']['name']}")
        mesh = data["poly_data"]
        max_range = np.max(mesh.points, axis=0)
        min_range = np.min(mesh.points, axis=0)
        lengths_range = max_range - min_range
        longest_range = np.max(lengths_range)
        scaled_points = (mesh.points - min_range) / longest_range
        remesh = PolyData(scaled_points, mesh.faces.copy())
        return remesh

    @staticmethod
    def mono_flipping(data):
        if VERBOSE: print(f"Flipping: {data['meta_data']['name']}")
        mesh = data["poly_data"]
        face_centers = mesh.cell_centers().points
        overall_signs = np.sum(np.sign(face_centers) * np.square(face_centers), axis=0)
        flipped_points = np.sign(overall_signs) * mesh.points
        remesh = PolyData(flipped_points, mesh.faces.copy())
        return remesh

    @staticmethod
    def mono_saving(data, target_path="processed_data"):
        print(f"Saving: {data['meta_data']['name']}")
        mesh = pv.PolyData(data["history"][-1]["data"]["vertices"], data["history"][-1]["data"]["faces"])
        target_directory = Path(f"{target_path}/{data['meta_data']['label']}")
        final_directory = target_directory / f"{data['meta_data']['name']}.ply"

        print(f"Writing {data['meta_data']['name']}.ply to {target_directory}")
        if not path.exists(target_directory):
            print(f"Creating path {target_directory} for saving {data['meta_data']['name']}")
            os.makedirs(target_directory)

        mesh.save(final_directory)
        print(f"{data['meta_data']['name']} was "
              f"{'successfully saved.'if path.exists(final_directory) else 'NOT saved!'}")
        return data

    @staticmethod
    def mono_run_pipeline(data):
        if not data: return False  # If user cancelled operation
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

        history = [{"op": step["op"], "data": {"vertices": step["data"].points, "faces": step["data"].faces}} for step
                   in history]

        copy_data = dict(data)
        copy_data.update({"data": history[-1]["data"]})
        print(f"Pipeline complete for {data['meta_data']['name']}")
        return dict(copy_data, history=history)

    def run_full_pipeline(self, max_num_items=None):
        num_full_data = len(self.reader.full_data)
        relevant_subset_of_data = self.reader.full_data[
                                  :min(max_num_items, num_full_data)] if max_num_items else self.reader.full_data
        num_data_being_processed = len(relevant_subset_of_data)
        normalization_data_generator = compute_normalization(self, relevant_subset_of_data)
        items_generator = tqdm(normalization_data_generator, total=num_data_being_processed)
        return list((self.mono_saving(item, self.target_path) for item in items_generator))


if __name__ == '__main__':
    norm = Normalizer(PSBDataset(DATA_PATH_DEBUG if DEBUG else DATA_PATH_PSB, class_file_path=CLASS_FILE))
    norm.run_full_pipeline(10 if DEBUG else None)
    print("Done")
