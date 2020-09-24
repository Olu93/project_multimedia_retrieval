import numpy as np
import pyacvd
from pyvista import PolyData

from reader import PSBDataset


class Normalizer:

    def __init__(self):
        self.num_avg_verts = 35000

        # self.reader = PSBDataset(search_path="D:\\Documents\\Programming\\Python\\project_multimedia_retrieval\\test_mesh")
        ## self.reader = PSBDataset(search_path="D:\\Downloads\\psb_v1\\benchmark\\db")
        self.reader = PSBDataset(search_path="C:\\Users\\chris\\OneDrive\\Dokumente\\Utrecht Uni Docs\\5.Period\\MS\\psb_v1\\benchmark\\db")
        self.reader.read()
        self.reader.load_files_in_memory()
        self.reader.convert_all_to_polydata()
        self.reader.compute_shape_statistics()
        self.full_data = self.reader.full_data
        self.history = []
        self.history.append([item["poly_data"] for item in self.full_data])
        # print(self.history[-1][0].bounds)

    #https://www.grasshopper3d.com/forum/topics/best-uniform-remesher-for-patterning-organic-suraces
    def uniform_remeshing(self):
        tmp_mesh = []
        remesh = None
        for mesh in self.history[-1]:
            data = mesh.clean()
            clus = pyacvd.Clustering(data)
            # if len(mesh.points) < 3500:
            while len(clus.mesh.points) < 10000:
                clus.subdivide(1)
            clus.cluster(10000)
            remesh = clus.create_mesh()
            tmp_mesh.append(remesh)
        self.history.append(tmp_mesh)

            # if mesh.n_faces < 9000:
            #     clust_mesh = pyacvd.Clustering(data)
            #     clust_mesh.subdivide(1)
            #     clust_mesh.cluster(10000)
            #     remesh = PolyData(clust_mesh.mesh.points, clust_mesh.mesh.faces)
            # # elif len(mesh.points) > self.num_avg_verts:
            # if mesh.n_faces > 11000:
            #     remesh = data.decimate(10000/mesh.n_faces)


    def center(self):
        tmp_mesh = []
        for mesh, full_mesh_stuff in zip(self.history[-1], self.full_data):
            remesh = PolyData(mesh.points.copy(), mesh.faces.copy())
            offset = full_mesh_stuff["bary_center"]
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
            # if (index % 100 == 0) and (index != 0):
            #     c_idx += 1
            # print(f"Writing m{index}.ply to data\\{c_idx}\\")
            # mesh.save(f"data\\{c_idx}\\m{index}.ply")
            print(f"Writing m{index}.ply to data")
            mesh.save(f"data\\m{index}.ply")


if __name__ == '__main__':
    norm = Normalizer()
    norm.scale_to_union()
    norm.center()
    norm.align()
    norm.uniform_remeshing()
    norm.save_dataset()
    print("Done")
