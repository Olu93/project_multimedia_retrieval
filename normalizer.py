import numpy as np
import pyacvd
from pyvista import PolyData
from reader import PSBDataset


class Normalizer:

    def __init__(self):
        self.num_avg_verts = 10000

        self.reader = PSBDataset(search_path="D:\\Downloads\\psb_v1\\benchmark\\db")
        self.reader.read()
        self.reader.load_files_in_memory()
        self.reader.convert_all_to_polydata()
        self.full_data = self.reader.full_data
        self.history = []
        self.history.append([item["poly_data"] for item in self.full_data])

    def uniform_remeshing(self):
        tmp_mesh = []
        remesh = None
        for mesh in self.history[-1]:
            data = mesh.clean()
            if len(mesh.points) < self.num_avg_verts:
                remesh = pyacvd.Clustering(data)
                remesh.subdivide(1)
            elif len(mesh.points) > self.num_avg_verts:
                remesh = data.decimate(0.7)  # Look a MAGIC NUMBER
            tmp_mesh.append(remesh)
        self.history.append(tmp_mesh)

    def center(self):
        tmp_mesh = []
        for mesh in self.history[-1]:
            remesh = PolyData(mesh.points, mesh.faces)
            offset = np.negative(remesh.center)
            remesh.translate(offset)
            tmp_mesh.append(remesh)
        self.history.append(tmp_mesh)

    def align(self):
        tmp_mesh = []
        for mesh in self.history[-1]:
            A_cov = np.cov(mesh.points.T)
            eigenvalues, eigenvectors = np.linalg.eig(A_cov)
            biggest_idx = np.argsort(-eigenvalues)
            biggest_vec = eigenvectors[:, biggest_idx]
            new_points = np.dot(mesh.points, biggest_vec)
            remesh = PolyData(new_points, mesh.faces)
            tmp_mesh.append(remesh)
        self.history.append(tmp_mesh)

    def scale_to_union(self):
        tmp_mesh = []
        for mesh in self.history[-1]:
            max_range = np.max(mesh.points, axis=0)
            min_range = np.min(mesh.points, axis=0)
            lengths_range = max_range - min_range
            longest_range = np.max(lengths_range)
            scaled_points = (mesh.points - min_range) / longest_range
            remesh = PolyData(scaled_points, mesh.faces)
            tmp_mesh.append(remesh)
        self.history.append(tmp_mesh)

    def save_dataset(self):
        for index, mesh in enumerate(self.history[-1]):
            mesh.save(f"data\\m{index}.ply")


if __name__ == '__main__':
    norm = Normalizer()
    norm.center()
    norm.align()
    # FLIP
    norm.scale_to_union()
    norm.save_dataset()
    norm.uniform_remeshing()
    print("Done")
