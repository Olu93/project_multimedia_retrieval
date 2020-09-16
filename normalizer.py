import pyacvd
import pyvista as pv

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
                # self.full_data[i]["poly_data"].plot(color="w", show_edges="True")
                data = self.full_data[i]["poly_data"].clean()
                remesh = pyacvd.Clustering(data)
                remesh.subdivide(1)
                # remesh.mesh.plot(color="w", show_edges="True")
                self.full_normalized_data.append(remesh.mesh)
            elif len(self.full_data[i]["poly_data"].points) > self.num_avg_verts:
                self.full_data[i]["poly_data"].plot(color="w", show_edges="True")
                data = self.full_data[i]["poly_data"].clean()
                data = data.decimate(0.7)  # Look a MAGIC NUMBER
                data.plot(color="w", show_edges="True")
                self.full_normalized_data.append(data)

    def center(self):
        # TODO: Take the pivot-point of the model and shift it to a mutual center w.r.t scene coordinates
        pass

    def scale_to_union(self):
        for i, mesh in enumerate(self.full_normalized_data):
            cube = pv.Cube()
            # TODO: fit the mesh to unit cube

    def save_dataset(self):
        for i, mesh in enumerate(self.full_normalized_data):
            mesh.save(f"D:\\Documents\\Programming\\Python\\project_multimedia_retrieval\\data\\m{i}.ply")


if __name__ == '__main__':
    norm = Normalizer()
    norm.uniform_remeshing()
    norm.save_dataset()
