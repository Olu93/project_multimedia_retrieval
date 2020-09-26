import numpy as np
import pyacvd
from pyvista import PolyData
from os import path
import os
from reader import PSBDataset

from helper.config import DEBUG, DATA_PATH, CLASS_FILE



class Normalizer:
    def __init__(self):
        self.num_avg_verts = 35000
        self.reader = PSBDataset(DATA_PATH, class_file_path=CLASS_FILE)
        self.reader.read()
        self.reader.load_files_in_memory()
        self.reader.convert_all_to_polydata()
        self.reader.compute_shape_statistics()
        self.full_data = self.reader.full_data
        self.history = []
        self.history.append([item["poly_data"] for item in self.full_data])
        # print(self.history[-1][0].bounds)

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

    def center(self):
        print("Centering")
        tmp_mesh = []
        for mesh, full_mesh_stuff in zip(self.history[-1], self.full_data):
            remesh = PolyData(mesh.points.copy(), mesh.faces.copy())
            offset = full_mesh_stuff["bary_center"]
            remesh.translate(np.zeros_like(offset) - offset)
            tmp_mesh.append(remesh)
        self.history.append(tmp_mesh)

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

    def scale_to_union(self):
        print("Scaling")
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

    def save_dataset(self):
        for processed_mesh, data in zip(self.history[-1], self.full_data):
            # if (index % 100 == 0) and (index != 0):
            #     c_idx += 1
            target_directory = f"processed_data\\{data['meta_data']['label']}"
            print(f"Writing {data['meta_data']['name']}.ply to {target_directory}")
            if not path.exists(target_directory):
                os.mkdir(target_directory)
            processed_mesh.save(f"{target_directory}\\m{data['meta_data']['name']}.ply")
            # print(f"Writing m{index}.ply to data")
            # mesh.save(f"data\\m{index}.ply")


if __name__ == '__main__':
    norm = Normalizer()
    norm.scale_to_union()
    norm.center()
    norm.align()
    norm.uniform_remeshing()
    norm.save_dataset()
    print("Done")
