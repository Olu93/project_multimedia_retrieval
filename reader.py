from pathlib import Path
import pandas as pd
import numpy as np
import glob
from pprint import pprint
from plyfile import PlyData
import pyvista as pv
import os
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain

class DataSet:
    data_descriptors = []
    data = []

    def __init__(self, search_paths):
        data_folder = [Path(path) for path in search_paths]
        self.data_file_paths = list(chain(*[glob.glob(str(path), recursive=True) for path in data_folder]))

    def read(self):
        raise NotImplementedError  

    def load_files_in_memory(self):
        self.data = [self._load_ply(file["path"]) if not file["type"]==".off" else self._load_off(file["path"]) for file in self.data_descriptors]

    
    def _load_ply(self, file):
        ply_data = PlyData.read(os.open(file, "r"))
        vertices = np.array(ply_data["vertex"].data.tolist())
        faces = np.array(ply_data["face"]["vertex_indices"].tolist())
        faces = np.hstack([(np.ones(faces.shape[0])*faces.shape[1]).reshape(-1,1), faces]).flatten()
        return {"vertices":vertices, "faces":faces}

    def _load_off(self, file):
        off_data = pv.read(file)
        faces = off_data.cells
        vertices = np.array(off_data.points)
        return {"vertices":vertices, "faces":faces}


    def _extract_descr(self, file_path):
        raise NotImplementedError

    def show_class_histogram(self):
        pd_data = pd.DataFrame(self.data_descriptors)
        counts_list = list(Counter(pd_data["label"].astype(str).values).items())
        counts = pd.DataFrame([{"label": label, "counts": int(count)}  for label, count in counts_list])
        # counts = np.array()
        plt.bar(counts["label"], counts["counts"]) 
        plt.show()


class PSBDataset(DataSet):
    def __init__(self):
        search_paths = [Path("data/psb") / "**/*.off", Path("data/psb") / "**/*.ply"]
        super().__init__(search_paths)

    def read(self):
        self.data_descriptors = [self._extract_descr(file_path) for file_path in self.data_file_paths]
        pprint(self.data_descriptors)

    def _extract_descr(self, file_path):
        path = Path(file_path)
        label = str(path.parents[1]).split("/")[-1]
        file_name = path.stem
        file_type = path.suffix
        return {"label": int(label), "name": file_name, "type": file_type, "path":path.resolve()}

    

if __name__ == "__main__":
    dataset = PSBDataset()
    dataset.read()
    dataset.show_class_histogram()
    # dataset.load_files_in_memory()
    # print(dataset.data[0])
