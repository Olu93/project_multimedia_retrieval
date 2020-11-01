from helper.skeleton import extract_sillhouettes
from helper.config import CLASS_FILE, DATA_PATH_NORMED, DATA_PATH_NORMED_SUBSET
from reader import PSBDataset
import numpy as np
import random as r
from scipy.spatial import ConvexHull
from pyvista import PolyData
from pyvista import examples
import pyvista as pv
from tqdm import tqdm
import multiprocessing as mp
from feature_extractor import FeatureExtractor


# def prepare_image(img):
#     img_copy = np.ones_like(img)
#     img_copy[np.isnan(img)] = 0
#     return img_copy


# def extract_sillhouettes(mesh, normal):
#     p = pv.Plotter(
#         notebook=False,
#         off_screen=True,
#     )
#     projected = mesh.project_points_to_plane((0, 0, 0), normal=normal)
#     p.add_mesh(projected)
#     p.set_position(normal * 2)
#     p.render()
#     img = p.get_image_depth()
#     return prepare_image(img)


def extract_graphical_forms(mesh):
    normals = np.eye(3) * -1
    sillhouettes = (extract_sillhouettes(mesh, normal) for normal in normals)
    return list(sillhouettes)

def wrapper(data):
    poly_data = pv.PolyData(data["data"]["vertices"], data["data"]["faces"])
    return extract_graphical_forms(poly_data)

# us_map = pv.read('trash/m1693.ply')
# bridge = pv.read('trash/m1785.ply')
def run(data):
    return extract_graphical_forms(us_map if data == 0 else bridge)

if __name__ == "__main__":
    FE = FeatureExtractor(PSBDataset(DATA_PATH_NORMED, class_file_path=CLASS_FILE))
    # print(len(FE.reader.full_data))
    # FE.run_full_pipeline_slow()
    # FE.run_full_pipeline()
    # with mp.Pool(5) as pool:
    # data = [r.choice([0, 1]) for _ in range(2000)]
    results = list(mp.Pool(5).imap_unordered(wrapper, tqdm(FE.reader.full_data)))
    print(len(results))

    # print(img.max())
    # print(img.min())
