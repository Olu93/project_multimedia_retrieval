from reader import PSBDataset, DataSet
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
from helper.config import DEBUG, DATA_PATH_PSB, DATA_PATH_NORMED, DATA_PATH_DEBUG, CLASS_FILE
import pyvista as pv
import itertools
from collections import Counter
import pandas as pd
import scipy

if __name__ == "__main__":
    origFaceareas = []
    normedFaceareas = []

    origDB = PSBDataset(DATA_PATH_PSB, class_file_path=CLASS_FILE)
    origDB.read()
    origDB.load_files_in_memory()

    normedDB = PSBDataset(DATA_PATH_NORMED, class_file_path=CLASS_FILE)
    normedDB.read()
    normedDB.load_files_in_memory()

    origFaceareas = [DataSet.get_only_cell_areas(mesh.get('data').get('vertices'), mesh.get('data').get('faces')) for mesh in origDB.full_data]
    origFaceareas = list(itertools.chain(*origFaceareas))
    origFaceareas_sub = pd.DataFrame(origFaceareas, columns=['fa'])
    origFaceareas_sub.to_csv("origFaceAreas.csv")
    origFaceareas_sub = origFaceareas_sub.sort_values(by='fa')
    origFaceareas_sub = origFaceareas_sub[origFaceareas_sub['fa'] > 0]
    origFaceareas_sub = origFaceareas_sub.iloc[int(.5 * len(origFaceareas_sub)):int(.80 * len(origFaceareas_sub))]
    origFaceareas_sub['fa'] = origFaceareas_sub['fa'] * 10000
    origmin = min(origFaceareas_sub['fa'])
    origmax = max(origFaceareas_sub['fa'])

    origbins = np.linspace(origmin, origmax, 10)
    np.set_printoptions(suppress=True)
    origbins_round = np.around(origbins, 2)
    origindices = np.digitize(origFaceareas_sub['fa'], origbins_round)
    count_dict_orig = dict(sorted(Counter(origindices).items()))
    count_dict_without_holes_orig  = {idx: count_dict_orig[idx] if idx in count_dict_orig.keys() else 0 for idx in
                                range(1, 11)}
    result_orig = np.array(list(count_dict_without_holes_orig.values()))

    out_orig = pd.cut(origFaceareas_sub['fa'], bins=origbins_round.tolist(), include_lowest=True)
    out_norm_orig = out_orig.value_counts(sort=False, normalize=True)
    ax = out_norm_orig.plot.bar(rot=0, color="b", figsize=(15, 4))

    normedFaceareas = [DataSet.get_only_cell_areas(mesh.get('data').get('vertices'), mesh.get('data').get('faces')) for mesh in normedDB.full_data]
    normedFaceareas = list(itertools.chain(*normedFaceareas))
    normedFaceareas = pd.DataFrame(normedFaceareas, columns=['fa'])
    normedFaceareas.to_csv("normedFaceAreas.csv")
    normedFaceareas_sub = normedFaceareas.sort_values(by='fa')
    normedFaceareas_sub = normedFaceareas_sub[normedFaceareas_sub['fa'] > 0]
    normedFaceareas_sub = normedFaceareas_sub.iloc[int(.5 * len(normedFaceareas_sub)):int(.80 * len(normedFaceareas_sub))]
    normedFaceareas_sub['fa'] = normedFaceareas_sub['fa'] * 10000
    normedmin = min(normedFaceareas_sub['fa'])
    normedmax = max(normedFaceareas_sub['fa'])

    normedbins = np.linspace(normedmin, normedmax, 10)
    normedbins_round = np.around(normedbins, 2)
    normedindices = np.digitize(normedFaceareas_sub['fa'], normedbins_round)
    count_dict_normed = dict(sorted(Counter(normedindices).items()))
    count_dict_without_holes_normed = {idx: count_dict_normed[idx] if idx in count_dict_normed.keys() else 0 for idx in
                                     range(1, 11)}
    result_normed = np.array(list(count_dict_without_holes_normed.values()))

    out_normed = pd.cut(normedFaceareas_sub['fa'], bins=normedbins_round.tolist(), include_lowest=True)
    out_norm_normed = out_normed.value_counts(sort=False, normalize=True)
    ax = out_norm_normed.plot.bar(rot=0, color="b", figsize=(15, 4))

    print(" ")
