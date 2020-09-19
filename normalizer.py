import numpy as np
import pyacvd

from reader import PSBDataset

class Normalizer:

    def __init__(self):
        self.num_avg_verts = 10000

        self.reader = PSBDataset()
        self.reader.read()
        self.reader.load_files_in_memory()
        self.reader.compute_shape_statistics()
        self.reader.convert_all_to_polydata()
        self.full_data = self.reader.full_data
        self.full_normalized_data = []

    def uniform_remeshing(self):
        for i in range(len(self.full_data)):
            if len(self.full_data[i]["poly_data"].points) < self.num_avg_verts:
                data = self.full_data[i]["poly_data"].clean()
                remesh = pyacvd.Clustering(data)
                remesh.subdivide(1)
                self.full_normalized_data.append(remesh.mesh)
            elif len(self.full_data[i]["poly_data"].points) > self.num_avg_verts:
                data = self.full_data[i]["poly_data"].clean()
                data = data.decimate(0.7)  # Look a MAGIC NUMBER
                self.full_normalized_data.append(data)

    def center(self):
        for mesh in self.full_normalized_data:
            offset = np.negative(mesh.center)
            mesh.translate(offset)

    def align(self):
        for mesh in self.full_normalized_data:
            A_cov = np.cov(mesh.points.T)
            eigenvalues, eigenvectors = np.linalg.eig(A_cov)
            biggest_idx = np.argsort(-eigenvalues)
            biggest_vec = eigenvectors[:, biggest_idx]
            new_points = np.dot(mesh.points, biggest_vec)
            mesh.points = new_points

    def scale_to_union(self):
        for mesh in self.full_normalized_data:
            max_range = np.max(mesh.points, axis=0)
            min_range = np.min(mesh.points, axis=0)
            lengths_range = max_range - min_range
            longest_range = np.max(lengths_range)
            scaled_points = (mesh.points - min_range) / longest_range
            mesh.points = scaled_points

    def save_dataset(self):
        for index, mesh in enumerate(self.full_normalized_data):
            mesh.save(f"data\\m{index}.ply")


if __name__ == '__main__':
    norm = Normalizer()
    norm.center()
    norm.align()
    #FLIP
    norm.scale_to_union()
    norm.save_dataset()
    norm.uniform_remeshing()
    print("Done")
