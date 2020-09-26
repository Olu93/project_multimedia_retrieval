#%%
import pygmsh
import pyvista as pv
from pyvista import examples
import meshio
from reader import PSBDataset
import io
import pyacvd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rescale(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

# %%
# self.reader = PSBDataset(search_path="D:\\Documents\\Programming\\Python\\project_multimedia_retrieval\\test_mesh")
## self.reader = PSBDataset(search_path="D:\\Downloads\\psb_v1\\benchmark\\db")
readerOrig = PSBDataset(
    search_path="C:\\Users\\chris\\OneDrive\\Dokumente\\Utrecht Uni Docs\\5.Period\\MS\\psb_v1\\benchmark\\db")
readerOrig.read()
readerOrig.load_files_in_memory()
readerOrig.convert_all_to_polydata()
readerOrig.compute_shape_statistics()
stats = readerOrig.all_statistics
cell_std_orig = stats.cell_area_std
# print(stats.cell_area_mean)
plt.hist(rescale(cell_std_orig), bins=20)


readerNorm = PSBDataset(
    search_path="data")
readerNorm.read()
readerNorm.load_files_in_memory()
readerNorm.convert_all_to_polydata()
readerNorm.compute_shape_statistics()
stats = readerNorm.all_statistics
cell_std_norm = stats.cell_area_std
plt.hist(rescale(cell_std_norm), bins=20, alpha=.5)
# print(stats.cell_area_mean)



# from matplotlib.ticker import PercentFormatter
# plt.gca().xaxis.set_major_formatter(PercentFormatter(1))

plt.show()