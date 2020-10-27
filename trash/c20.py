import numpy as np
import random as r
from scipy.spatial import ConvexHull
from pyvista import PolyData
from pyvista import examples
import pyvista as pv
from tqdm import tqdm
import multiprocessing as mp



def prepare_image(img):
    img_copy = np.ones_like(img)
    img_copy[np.isnan(img)] = 0
    return img_copy


def extract_sillhouettes(mesh, normal):
    p = pv.Plotter(
        notebook=False,
        off_screen=True,
    )
    projected = mesh.project_points_to_plane((0, 0, 0), normal=normal)
    p.add_mesh(projected)
    p.set_position(normal * 2)
    p.render()
    img = p.get_image_depth()
    return prepare_image(img)


def extract_graphical_forms(mesh):
    normals = np.eye(3) * -1
    sillhouettes = (extract_sillhouettes(mesh, normal) for normal in normals)
    return list(sillhouettes)

us_map = pv.read('trash/m1693.ply')
bridge = pv.read('trash/m1785.ply')
def run(data):
    return extract_graphical_forms(us_map if data == 0 else bridge)

if __name__ == "__main__":
    pool = mp.Pool(5)
    data = [r.choice([0, 1]) for _ in range(2000)]
    list(pool.imap_unordered(run, tqdm(data), chunksize=1))

    # print(img.max())
    # print(img.min())
