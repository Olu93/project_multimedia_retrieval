import numpy as np
import pyacvd
import pyvista as pv
import numpy as np

from reader import PSBDataset


def unit_vector(vec):
    unit = vec / np.linalg.norm(vec)
    return unit


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
        # TODO: Take the pivot-point of the model and shift it to a mutual center w.r.t scene coordinates
        for mesh in self.full_normalized_data:
            print(f"Before: {mesh.center}")
            offset = np.negative(mesh.center)
            mesh.translate(offset)
            print(f"After: {mesh.center}")

    def align(self):
        # TODO: Align to unit vector
        for mesh in self.full_normalized_data:
            pass

    def scale_to_union(self, mesh):
        max_range = np.max(mesh.points, axis=0)
        min_range = np.min(mesh.points, axis=0)
        lengths_range = max_range - min_range
        longest_range = np.max(lengths_range)
        scaled_points = (mesh.points - min_range) / longest_range
        mesh.points = scaled_points
        return mesh

    def save_dataset(self, mesh, index):
        mesh.save(f"data\\m{i}.ply")


if __name__ == '__main__':
    norm = Normalizer()
    norm.uniform_remeshing()
    norm.center()
    print("Done")
    # norm.save_dataset()
