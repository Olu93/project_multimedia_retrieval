import pathlib
from helper.misc import make_bins, normalize, remove_outlier
from reader import PSBDataset, DataSet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helper.config import DEBUG, DATA_PATH_PSB, DATA_PATH_NORMED, DATA_PATH_DEBUG, CLASS_FILE
import pyvista as pv
import itertools
from collections import Counter
import pandas as pd
import scipy


def construct_descriptor_string(ndarray):
    textstr_1 = "Mean " + f"({ndarray.mean():.2f})"
    textstr_2 = "Median " + f"({np.median(ndarray):.2f})"
    textstr_3 = "Std.dev " + f"({np.std(ndarray):.2f})"
    textstr_4 = "Std.err " + f"({scipy.stats.sem(ndarray):.2f})"
    stat_string = ", ".join([textstr_1, textstr_2, textstr_3, textstr_4])
    return stat_string


def construct_descriptor_string_scientific(ndarray):
    textstr_1 = "Mean " + f"({ndarray.mean():.2e})"
    textstr_2 = "Median " + f"({np.median(ndarray):.2e})"
    textstr_3 = "Std.dev " + f"({np.std(ndarray):.2e})"
    textstr_4 = "Std.err " + f"({scipy.stats.sem(ndarray):.2e})"
    stat_string = ", ".join([textstr_1, textstr_2, textstr_3, textstr_4])
    return stat_string


if __name__ == "__main__":
    origFaceareas = []
    normedFaceareas = []

    if not pathlib.Path("stats/orig_stats.csv").exists():
        origDB = PSBDataset(DATA_PATH_PSB, class_file_path=CLASS_FILE)
        origDB.run_full_pipeline()
        origDB.compute_shape_statistics()
        origDB.save_statistics(stats_path="stats", stats_filename="orig_stats.csv")

    if not pathlib.Path("stats/norm_stats.csv").exists():
        normedDB = PSBDataset(DATA_PATH_NORMED, class_file_path=CLASS_FILE)
        normedDB.run_full_pipeline()
        normedDB.compute_shape_statistics()
        normedDB.save_statistics(stats_path="stats", stats_filename="norm_stats.csv")

    orig_stats = pd.read_csv("stats/orig_stats.csv")
    norm_stats = pd.read_csv("stats/norm_stats.csv")

    orig_stats_cleansed = remove_outlier(orig_stats, "cell_area_mean")
    norm_stats_cleansed = remove_outlier(norm_stats, "cell_area_mean")

    orig_stats_cell_area_means = normalize(orig_stats_cleansed.cell_area_mean)
    norm_stats_cell_area_means = normalize(norm_stats_cleansed.cell_area_mean)

    print(("Statistical descriptors of the orig. dist. pre normalisation: " + construct_descriptor_string_scientific(orig_stats_cleansed.cell_area_mean.values).lower()))
    print(("Statistical descriptors of the norm. dist. pre normalisation: " + construct_descriptor_string_scientific(norm_stats_cleansed.cell_area_mean.values).lower()))
    print(("Statistical descriptors of the orig. dist. post normalisation: " + construct_descriptor_string(orig_stats_cell_area_means).lower()))
    print(("Statistical descriptors of the norm. dist. post normalisation: " + construct_descriptor_string(norm_stats_cell_area_means).lower()))

    # props = dict(boxstyle='square', facecolor='white', alpha=1)

    n_bins = 20

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist([orig_stats_cell_area_means, norm_stats_cell_area_means], bins=n_bins, label=['Original', 'Normalized'])
    # ax.text(0.05, 0.95, joined_box_string, fontsize=14, transform=ax.transAxes, verticalalignment='top', bbox=props)
    ax.legend(loc='upper right')
    ax.set_title("Mean Cell Area Distribution")
    plt.tight_layout()
    plt.savefig("figs/fig_cell_area_distribution_variant_1.png")
    plt.show()
    plt.close()

    n_bins = 100

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(orig_stats_cell_area_means, bins=n_bins, alpha=0.5, label='Original')
    ax.hist(norm_stats_cell_area_means, bins=n_bins, alpha=0.5, label='Normalized')
    ax.legend(loc='upper right')
    ax.set_title("Mean Cell Area Distribution")
    plt.tight_layout()
    plt.savefig("figs/fig_cell_area_distribution_variant_2.png")
    plt.show()
    plt.close()

    print("done!")