import numpy as np
import pyacvd
from pyvista import PolyData

from reader import PSBDataset


class Normalizer:

    def __init__(self):
        self.num_avg_verts = 35000

        self.reader = PSBDataset(search_path="D:\\Documents\\Programming\\Python\\project_multimedia_retrieval\\test_mesh")
        ## self.reader = PSBDataset(search_path="D:\\Downloads\\psb_v1\\benchmark\\db")
        self.reader.read()
        self.reader.load_files_in_memory()
        self.reader.convert_all_to_polydata()
        self.full_data = self.reader.full_data
        self.history = []
        self.history.append([item["poly_data"] for item in self.full_data])
        # print(self.history[-1][0].bounds)

    def uniform_remeshing(self):
        tmp_mesh = []
        remesh = None
        for mesh in self.history[-1]:
            data = mesh.clean()
            if len(mesh.points) < 3500:
                clust_mesh = pyacvd.Clustering(data)
                clust_mesh.subdivide(1)
                remesh = PolyData(clust_mesh.mesh.points, clust_mesh.mesh.faces)
            elif len(mesh.points) > self.num_avg_verts:
                remesh = data.decimate(0.7)
            tmp_mesh.append(remesh)
        self.history.append(tmp_mesh)

    def center(self):
        tmp_mesh = []
        for mesh in self.history[-1]:
            remesh = PolyData(mesh.points.copy(), mesh.faces.copy())
            offset = mesh.center
            remesh.translate(np.zeros_like(offset) - offset)
            tmp_mesh.append(remesh)
        self.history.append(tmp_mesh)
        # print(self.history[-1][0].bounds)

    def align(self):
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
        # print(self.history[-1][0].bounds)
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
        # print(self.history[-1][0].bounds)

    def save_dataset(self):
        c_idx = 0
        for index, mesh in enumerate(self.history[-1]):
            if (index % 100 == 0) and (index != 0):
                c_idx += 1
            print(f"Writing m{index}.ply to data\\{c_idx}\\")
            mesh.save(f"data\\{c_idx}\\m{index}.ply")


if __name__ == '__main__':
    norm = Normalizer()
    norm.scale_to_union()
    norm.center()
    norm.align()
    norm.uniform_remeshing()
    norm.save_dataset()
    print("Done")
